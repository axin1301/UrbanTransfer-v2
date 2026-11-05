"""K7 E1/E2: task-description-to-adapter hypernetwork.

E1: short task description -> generated low-rank visual adapter.
E2: same adapter with an additive, gated source->target relation message.
The relation is bound to source-indicator identity upstream (z1_relation.py).

This is a deliberately controlled K7 prototype, NOT Hyper-X/HINT reproduction.
Both encoders are frozen; only the parameter generator and readout train.
"""
import torch
from torch import nn
import torch.nn.functional as F


class K7HyperRegressor(nn.Module):
    """Generate a rank-r residual adapter for each (image, indicator) pair."""

    def __init__(self, visual_dim, semantic_dim, hidden=128, rank=8,
                 use_relation=False):
        super().__init__()
        if hidden < 8 or rank < 1:
            raise ValueError("Invalid K7 dimensions")
        self.hidden = hidden
        self.rank = rank
        self.use_relation = use_relation
        self.visual = nn.Sequential(
            nn.LayerNorm(visual_dim + 4),
            nn.Linear(visual_dim + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.task = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        if use_relation:
            self.relation = nn.Sequential(
                nn.LayerNorm(semantic_dim),
                nn.Linear(semantic_dim, hidden),
                nn.Tanh(),
            )
            # Initialize close to E1, but trainable; not an oracle relation.
            self.relation_gate = nn.Parameter(torch.tensor(-1.0))
        self.generator = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * hidden * rank),
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self.bias = nn.Linear(hidden, 1)
        self.adapter_gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, images, context, semantics, relations=None):
        if images.ndim != 2 or context.shape != (len(images), 4):
            raise ValueError("Expected visual [batch,d] and context [batch,4]")
        if semantics.ndim != 2 or len(semantics) != len(images):
            raise ValueError("Expected one semantic vector per image")
        if self.use_relation:
            if relations is None or relations.shape != semantics.shape:
                raise ValueError("K7 E2 needs paired relation embeddings")
        elif relations is not None:
            raise ValueError("K7 E1 must not use relation embeddings")
        x = self.visual(torch.cat([images.float(), context.float()], dim=-1))
        task = self.task(F.normalize(semantics.float(), dim=-1))
        if self.use_relation:
            rel = self.relation(F.normalize(relations.float(), dim=-1))
            task = task + torch.sigmoid(self.relation_gate) * rel
        params = self.generator(task)
        a, b = params.chunk(2, dim=-1)
        # Batch-specific A [batch, hidden, rank], B [batch, rank, hidden].
        # Tanh and rank scaling prevent parameter-generated explosion.
        a = torch.tanh(a.reshape(-1, self.hidden, self.rank))
        b = torch.tanh(b.reshape(-1, self.rank, self.hidden))
        z = torch.bmm(x.unsqueeze(1), a)
        delta = torch.bmm(z, b).squeeze(1) / self.rank
        adapted = x + self.adapter_gain * delta
        return (self.readout(adapted) + self.bias(task)).squeeze(-1)
