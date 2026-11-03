"""K6 unified inference: frozen K4 bank, K5-B2 region composer, K5-A transfer,
and independent source-only scale routers.

Fixed, auditable ablation configurations. No outer heldout target statistics
or labels can enter the prediction path. Model selection must not use
development labels on a per-task basis.
"""
import numpy as np
import torch
from torch import nn
from .ktrans_k5c import IndependentScaleRouter
from .ktrans_k5d import MeasurementAwareScaleRouter

VARIANTS={
    'k4':('semantic',False,'semantic'),
    'k5a':('transfer',False,'semantic'),
    'router_only':('regional',False,'semantic'),
    'adapter_only':('semantic',True,'semantic'),
    'k5b2':('regional',True,'semantic'),
    'k5c':('regional',True,'k5c'),
    'k5d':('regional',True,'k5d'),
    'k6_full':('regional_transfer',True,'k5d'),
}

def normalize_route(weights):
    w=np.asarray(weights,dtype=float)
    if w.ndim!=1 or not np.all(np.isfinite(w)) or np.any(w<0) or w.sum()<=0:
        raise ValueError('Invalid expert routing weights')
    return w/w.sum()

def combine_prior(regional,semantic,transfer):
    """Add a source-only transfer log-prior correction to region gate."""
    ratio=transfer.clamp_min(1e-8)/semantic.clamp_min(1e-8)
    w=regional*ratio
    return w/w.sum(dim=-1,keepdim=True)

class UnifiedKTransMoE(nn.Module):
    def __init__(self,composer,vectors,source_stats,heldout_task,transfer_weights):
        super().__init__()
        self.composer=composer
        self.bank=composer.bank
        self.target=heldout_task
        if heldout_task in self.bank.source_tasks or heldout_task not in self.bank.task_to_all:
            raise ValueError('Heldout task must be a known-but-unsupervised task')
        if tuple(sorted(source_stats))!=self.bank.source_tasks:
            raise ValueError('Source moments/task list mismatch')
        self.register_buffer('transfer',torch.as_tensor(
            normalize_route(transfer_weights),dtype=torch.float32))
        if self.transfer.numel()!=len(self.bank.source_tasks):
            raise ValueError('Transfer weights must match source expert count')
        self.k5c=IndependentScaleRouter(vectors,source_stats)
        self.k5d=MeasurementAwareScaleRouter(vectors,source_stats)
        self.k5c_moments=self.k5c.predict(heldout_task)
        self.k5d_moments=self.k5d.predict(heldout_task)
        self.eval()

    def forward_variant(self,batch,variant):
        if variant not in VARIANTS:raise ValueError(f'Unknown K6 variant {variant}')
        route_type,adapter,scale_type=VARIANTS[variant]
        names=batch['task_name']
        if any(t!=self.target for t in names):
            if all(t in self.bank.task_to_source for t in names):
                # Known-task UCSI uses the standard trained source expert;
                # this is held fixed across all zero-shot variants.
                return self.bank(batch)
            raise ValueError('Mixed seen/unseen task batches are unsupported')
        original_strength=self.composer.adapter_strength
        try:
            self.composer.adapter_strength=original_strength if adapter else 0.
            _,aux=self.composer(batch,return_aux=True)
        finally:
            self.composer.adapter_strength=original_strength
        regional=aux['weights']
        prior=aux['prior']
        target_transfer=self.transfer.to(device=regional.device,dtype=regional.dtype)
        if route_type=='semantic':w=prior
        elif route_type=='transfer':w=target_transfer.unsqueeze(0).expand_as(prior)
        elif route_type=='regional':w=regional
        elif route_type=='regional_transfer':
            w=combine_prior(regional,prior,target_transfer)
        else:raise AssertionError(route_type)
        spatial=(w*aux['expert_outputs']).sum(-1)
        if scale_type=='semantic':
            mu=(prior@self.bank.source_means)
            sigma=(prior@self.bank.source_stds)
        else:
            mean,std=self.k5c_moments if scale_type=='k5c' else self.k5d_moments
            mu=spatial.new_full((len(spatial),),mean)
            sigma=spatial.new_full((len(spatial),),std)
        return mu+sigma*spatial
