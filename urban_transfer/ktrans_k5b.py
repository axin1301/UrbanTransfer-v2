"""K5-B lightweight region × task conditioned routing and residual adaptation.

Freeze an existing K4 normalized expert bank. Train ONLY a small router and
shared-hidden residual adapter using outer source-task training labels. During
source-task meta-training each task's own expert is masked, preventing trivial
identity routing. Never train on the outer held-out task or held-out cities.
"""
import torch
from torch import nn
import torch.nn.functional as F

class RegionTaskComposer(nn.Module):
    def __init__(self,bank,task_vectors,rank=8,route_strength=1.,adapter_strength=.1,adapter_cap=None):
        super().__init__()
        self.bank=bank
        for p in self.bank.parameters():p.requires_grad_(False)
        self.bank.eval()
        self.source_tasks=bank.source_tasks
        self.task_names=bank.task_names
        self.source_index=bank.task_to_source
        self.all_index=bank.task_to_all
        self.n=len(self.source_tasks)
        self.rank=rank
        self.route_strength=route_strength
        self.adapter_strength=adapter_strength
        self.adapter_cap=adapter_cap
        vec=F.normalize(torch.stack([task_vectors[t].float() for t in self.task_names]),dim=-1)
        src=F.normalize(torch.stack([task_vectors[t].float() for t in self.source_tasks]),dim=-1)
        self.register_buffer('task_signature',vec@src.T)
        h=bank.shared[-2].out_features if isinstance(bank.shared[-2],nn.Linear) else None
        if h is None:raise ValueError('Cannot infer bank shared dimension')
        # Source similarity signature (directional input through task identity),
        # 4 geography inputs and 4 summary signals from image-conditioned h.
        self.summary=nn.Linear(h,4,bias=False)
        self.route=nn.Sequential(nn.Linear(self.n+8,32),nn.GELU(),nn.Linear(32,self.n))
        self.adapter_down=nn.Linear(h,rank,bias=False)
        self.adapter_gate=nn.Sequential(nn.Linear(self.n+4,32),nn.GELU(),nn.Linear(32,rank))
        self.adapter_up=nn.Linear(rank,h,bias=False)
        nn.init.zeros_(self.route[-1].weight);nn.init.zeros_(self.route[-1].bias)
        nn.init.zeros_(self.adapter_up.weight)  # bias=False

    def train(self,mode=True):
        super().train(mode)
        self.bank.eval()
        return self

    def forward(self,batch,exclude_own=False,return_aux=False):
        names=batch['task_name']
        device=batch['image'].device
        indices=torch.tensor([self.all_index[t] for t in names],device=device)
        context=batch['context'].float()
        with torch.no_grad():
            h=self.bank.shared(torch.cat([batch['image'].float(),context],-1))
        sig=self.task_signature[indices]
        gate=torch.tanh(self.adapter_gate(torch.cat([sig,context],-1)))
        residual=self.adapter_up(self.adapter_down(h)*gate)
        if self.adapter_cap is not None:
            if self.adapter_cap<=0:raise ValueError('adapter_cap must be positive')
            # Bound each residual coordinate without any target-label statistics.
            residual=self.adapter_cap*torch.tanh(residual/self.adapter_cap)
        h_adapt=h+self.adapter_strength*residual
        z=torch.cat([expert(h_adapt) for expert in self.bank.experts],dim=-1)
        if return_aux and self.adapter_cap is not None:
            with torch.no_grad():
                z_frozen=torch.cat([expert(h) for expert in self.bank.experts],dim=-1)
            preserve=(z-z_frozen).square().mean()
        else:
            preserve=z.new_zeros(())
        features=torch.cat([sig,context,self.summary(h)],-1)
        delta=self.route(features)
        prior=self.bank.routing[indices].clamp_min(1e-8)
        logits=prior.log()+self.route_strength*delta
        if exclude_own:
            own=torch.tensor([self.source_index.get(t,-1) for t in names],device=device)
            if (own<0).any():raise ValueError('Episodic training expects source tasks only')
            logits=logits.masked_fill(F.one_hot(own,self.n).bool(),-1e4)
        weights=F.softmax(logits,dim=-1)
        spatial=(weights*z).sum(-1)
        # Training targets use source-only moments for their respective known
        # indicator; unseen tasks use the *frozen* semantic scale router.
        if exclude_own:
            ids=torch.tensor([self.source_index[t] for t in names],device=device)
            mean=self.bank.source_means[ids]
            std=self.bank.source_stds[ids]
        else:
            mean=prior@self.bank.source_means
            std=prior@self.bank.source_stds
        prediction=mean+std*spatial
        if return_aux:
            return prediction,{'weights':weights,'prior':prior,'residual':residual,'spatial':spatial,'preserve_loss':preserve}
        return prediction
