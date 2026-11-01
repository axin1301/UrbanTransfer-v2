"""K3-A standardized source experts with source-only zero-shot scale inference.

Each expert predicts a standardized source-task spatial pattern. Seen tasks
reconstruct their original target using training-only source moments.
For an unseen task, the *same* routing weights combine standardized expert
patterns and infer output moments from source-task moments. No unseen target
labels are consulted.
"""
import numpy as np
import torch
import torch.nn.functional as F
from .ktrans_experts import SemanticExpertBank


def source_moments(training_df, source_tasks, minimum_std=1e-3):
    """Moments computed ONLY from the meta-train city/task subset."""
    moments={}
    for task in sorted(source_tasks):
        y=training_df.loc[training_df.indicator_column==task,'indicator_value_log1p'].to_numpy(dtype=np.float64)
        if len(y)<2 or not np.isfinite(y).all():
            raise ValueError(f'Insufficient/invalid training labels: {task}')
        moments[task]=(float(y.mean()),float(max(y.std(),minimum_std)))
    return moments


class NormalizedExpertBank(SemanticExpertBank):
    def __init__(self,input_dim,task_vectors,source_tasks,source_stats,hidden_dim=128,temperature=.15):
        super().__init__(input_dim,task_vectors,source_tasks,hidden_dim,temperature)
        if set(source_stats)!=set(self.source_tasks):
            raise ValueError('Source moments must exactly match supervised source tasks')
        self.scale_routing_mode='shared'
        self.register_buffer('source_means',torch.tensor([source_stats[t][0] for t in self.source_tasks],dtype=torch.float32))
        self.register_buffer('source_stds',torch.tensor([source_stats[t][1] for t in self.source_tasks],dtype=torch.float32))

    def set_scale_routing_mode(self,mode='shared'):
        if mode not in ('shared','semantic','uniform'):
            raise ValueError(f'Unknown scale routing mode: {mode}')
        self.scale_routing_mode=mode

    def forward(self,batch,knowledge=None):
        h=self.shared(torch.cat([batch['image'].float(),batch['context'].float()],dim=-1))
        z=torch.cat([expert(h) for expert in self.experts],dim=1)
        device=z.device
        names=batch['task_name']
        idx=torch.tensor([self.task_to_all[t] for t in names],device=device)
        if self.routing_mode=='uniform':
            route=torch.full_like(z,1./len(self.source_tasks))
        elif self.routing_mode=='random':
            route=self.random_routing[idx].clone()
        else:
            route=self.routing[idx].clone()
        # Supervised tasks are reconstructed in their own log1p units.
        for row,name in enumerate(names):
            if name in self.task_to_source:
                route[row]=F.one_hot(torch.tensor(self.task_to_source[name],device=device),
                        num_classes=len(self.source_tasks)).to(route.dtype)
        # Spatial shape and task scale are explicitly separate.
        spatial=(route*z).sum(-1)
        if self.scale_routing_mode=='shared':
            scale_route=route.clone()
        elif self.scale_routing_mode=='uniform':
            scale_route=torch.full_like(route,1./len(self.source_tasks))
        else:
            scale_route=self.routing[idx].clone()
        # Use a distinct tensor: reassigning rows of an alias after (route*z)\n        # invalidates the saved route needed by MulBackward.\n        # Seen-task predictions always use their own task moments.
        for row,name in enumerate(names):
            if name in self.task_to_source:
                scale_route[row]=route[row]
        mean=scale_route@self.source_means
        std=scale_route@self.source_stds
        return mean+std*spatial
