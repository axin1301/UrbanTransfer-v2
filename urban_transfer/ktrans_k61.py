"""K6.1-A: source-supervised gate between K5-A transfer and regional routes.

The transfer route is an immutable global anchor. The pre-trained K5-B2
composer provides regional expert weights with Adapter OFF. Only a tiny
gate is learned from OUTER TRAIN source-task rows, with the own source
expert excluded for each pseudo-unseen source task.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scripts.run_ktrans_k5a import features,softmax


def build_source_transfer_routes(vectors,source_tasks,source_stats,beta,temperature):
    """Pseudo-target source routes; never include a task's own expert."""
    if len(beta)!=4 or temperature<=0:
        raise ValueError('Expected K5-A four-feature ridge and positive temperature')
    routes=[]
    for target in source_tasks:
        others=[s for s in source_tasks if s!=target]
        scores=np.asarray([
            features(vectors[target],vectors[s],source_stats[s])@np.asarray(beta)
            for s in others])
        probs=softmax(scores,temperature)
        row=np.zeros(len(source_tasks),dtype=np.float32)
        for source,value in zip(others,probs):
            row[source_tasks.index(source)]=value
        routes.append(row)
    return torch.tensor(np.stack(routes))


class AnchoredRegionalGate(nn.Module):
    def __init__(self,composer,transfer_weights,source_routes):
        super().__init__()
        self.composer=composer
        for param in composer.parameters():
            param.requires_grad_(False)
        composer.eval()
        self.source_tasks=list(composer.source_tasks)
        self.n=len(self.source_tasks)
        tw=torch.as_tensor(transfer_weights,dtype=torch.float32)
        sr=torch.as_tensor(source_routes,dtype=torch.float32)
        if tw.shape!=(self.n,) or sr.shape!=(self.n,self.n):
            raise ValueError('Transfer weights have incompatible expert dimensions')
        if (tw<0).any() or (sr<0).any() or tw.sum()<=0 or (sr.sum(-1)<=0).any():
            raise ValueError('Transfer weights must be nonnegative distributions')
        if not torch.allclose(torch.diag(sr),torch.zeros(self.n),atol=1e-7):
            raise ValueError('Pseudo-unseen task routes must mask their own expert')
        self.register_buffer('target_transfer',tw/tw.sum())
        self.register_buffer('source_transfer',sr/sr.sum(-1,keepdim=True))
        # Signature n, geography 4, route disagreement 1, regional entropy 1.
        self.gate=nn.Linear(self.n+6,1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias,-2.)  # conservative 0.119 initial regional mix

    def train(self,mode=True):
        super().train(mode)
        self.composer.eval()
        return self

    def forward(self,batch,exclude_own=False,return_aux=False):
        device=batch['image'].device
        names=batch['task_name']
        if exclude_own:
            if any(t not in self.composer.source_index for t in names):
                raise ValueError('Gate training permits source tasks only')
            ids=torch.tensor([self.composer.source_index[t] for t in names],device=device)
            anchor=self.source_transfer[ids]
        else:
            if any(t in self.composer.source_index for t in names):
                raise ValueError('K6.1-A unseen evaluation requires heldout task rows')
            anchor=self.target_transfer.unsqueeze(0).expand(len(names),-1)
        # Adapter permanently OFF for K6.1-A; never mutate shared module
        # strength, so training/evaluation stay independent of K5-B2 adapter.
        if self.composer.adapter_strength!=0.:
            raise ValueError('K6.1-A composer adapter_strength must be set to zero')
        with torch.no_grad():
            _,aux=self.composer(batch,exclude_own=exclude_own,return_aux=True)
            regional=aux['weights']
            experts=aux['expert_outputs']
            indices=torch.tensor([self.composer.all_index[t] for t in names],device=device)
            signature=self.composer.task_signature[indices]
            disagreement=0.5*(anchor-regional).abs().sum(-1,keepdim=True)
            entropy=-(regional*regional.clamp_min(1e-8).log()).sum(-1,keepdim=True)/math.log(self.n)
            info=torch.cat([signature,batch['context'].float(),disagreement,entropy],dim=-1)
        strength=torch.sigmoid(self.gate(info)).squeeze(-1)
        weights=(1-strength.unsqueeze(-1))*anchor+strength.unsqueeze(-1)*regional
        spatial=(weights*experts).sum(-1)
        if exclude_own:
            own=torch.tensor([self.composer.source_index[t] for t in names],device=device)
            mean=self.composer.bank.source_means[own]
            std=self.composer.bank.source_stds[own]
        else:
            prior=aux['prior']
            mean=prior@self.composer.bank.source_means
            std=prior@self.composer.bank.source_stds
        pred=mean+std*spatial
        if return_aux:
            return pred,{'spatial':spatial,'gate':strength,'weights':weights,
                         'anchor':anchor,'regional':regional,
                         'expert_outputs':experts}
        return pred
