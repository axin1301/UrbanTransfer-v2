"""K2-A scale-conditioned regression with frozen semantic task vectors.

The calibration parameters come from the task embedding, never test labels.
"""
import math
import torch
from torch import nn
from .m3 import M3Predictor

class TaskScalePredictor(nn.Module):
    def __init__(self, cfg, input_dim, task_dim):
        super().__init__()
        self.backbone=M3Predictor(cfg,input_dim)
        self.scale_head=nn.Sequential(
            nn.LayerNorm(task_dim),
            nn.Linear(task_dim,64),
            nn.GELU(),
            nn.Linear(64,2),
        )
        # Identity initialization; all folds start with the same prediction
        # as a freshly initialized uncalibrated M3 backbone.
        nn.init.zeros_(self.scale_head[-1].weight)
        with torch.no_grad():
            self.scale_head[-1].bias[0]=0.
            self.scale_head[-1].bias[1]=math.log(math.expm1(1.0))
    def forward(self,batch,knowledge=None):
        base=self.backbone(batch,knowledge)
        bias,log_scale=self.scale_head(batch['task'].float()).unbind(dim=-1)
        scale=torch.nn.functional.softplus(log_scale).clamp(0.05,20.)
        return bias+scale*base
