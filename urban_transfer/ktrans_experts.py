"""K2-B1: supervised source-task spatial experts, fixed zero-shot semantic routing.

Each available source task trains its own regression expert. At inference, a
seen task uses its own expert; a held-out task combines source predictions
using frozen text cosine similarity. No held-out labels enter optimization.
"""
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class SemanticExpertBank(nn.Module):
    def __init__(self,input_dim,task_vectors,source_tasks,hidden_dim=128,temperature=0.15):
        super().__init__()
        if temperature<=0:raise ValueError('temperature must be positive')
        if not source_tasks:raise ValueError('Need source tasks')
        self.source_tasks=tuple(sorted(source_tasks))
        self.task_names=tuple(sorted(task_vectors))
        if not set(self.source_tasks).issubset(task_vectors):
            raise ValueError('Missing source-task embeddings')
        self.shared=nn.Sequential(nn.LayerNorm(input_dim+4),
            nn.Linear(input_dim+4,hidden_dim),nn.GELU(),
            nn.Linear(hidden_dim,hidden_dim),nn.GELU())
        self.experts=nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim,hidden_dim//2),nn.GELU(),
                          nn.Linear(hidden_dim//2,1))
            for _ in self.source_tasks])
        keys=F.normalize(torch.stack([task_vectors[t].float() for t in self.source_tasks]),dim=-1)
        queries=F.normalize(torch.stack([task_vectors[t].float() for t in self.task_names]),dim=-1)
        similarity=queries@keys.T
        # Convert potentially negative similarities to proper simplex weights.
        self.register_buffer('routing',F.softmax(similarity/temperature,dim=-1))
        self.routing_mode='semantic'
        self.task_to_source={t:i for i,t in enumerate(self.source_tasks)}
        self.task_to_all={t:i for i,t in enumerate(self.task_names)}
    def set_routing_mode(self,mode='semantic',seed=42):
        """Evaluation-only ablation: same trained experts, alternative unseen-task routes."""
        if mode not in ('semantic','uniform','random'):
            raise ValueError(f'Unknown routing mode: {mode}')
        self.routing_mode=mode
        if mode=='random':
            rng=np.random.default_rng(seed)
            weights=rng.dirichlet(np.ones(len(self.source_tasks)),size=len(self.task_names))
            self.register_buffer('random_routing',torch.tensor(weights,dtype=self.routing.dtype,
                              device=self.routing.device),persistent=False)

    def forward(self,batch,knowledge=None):
        x=torch.cat([batch['image'].float(),batch['context'].float()],dim=-1)
        h=self.shared(x)
        outputs=torch.cat([expert(h) for expert in self.experts],dim=1)
        names=batch['task_name']
        device=outputs.device
        indices=torch.tensor([self.task_to_all[t] for t in names],device=device)
        if self.routing_mode=='uniform':
            routes=torch.full((len(names),len(self.source_tasks)),1./len(self.source_tasks),
                              device=device,dtype=outputs.dtype)
        elif self.routing_mode=='random':
            routes=self.random_routing[indices].clone()
        else:
            routes=self.routing[indices].clone()
        # During training, only an example's own source-task expert receives
        # supervision. This also applies to seen tasks at validation.
        for row,name in enumerate(names):
            if name in self.task_to_source:
                routes[row]=F.one_hot(torch.tensor(self.task_to_source[name],device=device),
                                      num_classes=len(self.source_tasks)).to(routes.dtype)
        return (outputs*routes).sum(dim=-1)
