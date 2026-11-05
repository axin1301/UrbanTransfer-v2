"""Z1 models: shared semantic regression, conditional semantics, generated heads.

No per-task lookup embeddings or source expert selection. Unknown-task heads
are synthesized directly from frozen semantic embeddings.
"""
import torch
from torch import nn
import torch.nn.functional as F


class Z1Regressor(nn.Module):
    """B1: semantic FiLM; B2: image-conditioned semantics; B3: generated low-rank head."""

    def __init__(self, visual_dim, semantic_dim, mode="b1", hidden=128, rank=16):
        super().__init__()
        if mode not in ("b1", "b2", "b3"):
            raise ValueError("Z1 mode must be b1, b2 or b3")
        self.mode = mode
        self.visual = nn.Sequential(
            nn.LayerNorm(visual_dim + 4), nn.Linear(visual_dim + 4, hidden),
            nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.semantic = nn.Sequential(
            nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden),
            nn.GELU())
        self.film = nn.Linear(hidden, hidden * 2)
        if mode in ("b2", "b3"):
            self.contextualizer = nn.Sequential(
                nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden),
                nn.Tanh())
        if mode == "b3":
            self.generate = nn.Linear(hidden, rank)
            self.head_left = nn.Linear(hidden, rank, bias=False)
            self.base = nn.Linear(hidden, 1)
            self.output_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2),
                nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, images, context, semantics):
        if images.ndim != 2 or context.shape != (len(images), 4):
            raise ValueError("Expected [batch, visual_dim] and [batch,4] features")
        if semantics.ndim != 2 or len(semantics) != len(images):
            raise ValueError("Expected one fixed semantic vector per sample")
        visual = self.visual(torch.cat((images.float(), context.float()), -1))
        semantic = self.semantic(F.normalize(semantics.float(), dim=-1))
        if self.mode in ("b2", "b3"):
            semantic = semantic + self.contextualizer(torch.cat((visual, semantic), -1))
        gamma, beta = self.film(semantic).chunk(2, -1)
        fused = visual * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
        if self.mode == "b3":
            # Low-rank coefficients generated from task semantics, not a source head index.
            return (self.base(fused).squeeze(-1) +
                    self.output_scale * (self.head_left(fused) *
                                         torch.tanh(self.generate(semantic))).sum(-1))
        return self.head(fused).squeeze(-1)
