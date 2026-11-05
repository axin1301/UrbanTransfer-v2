"""K7 M2: two directed text-edge tokens -> source-specific visual adapters.

No averaging of edge embeddings. Each edge retains its source identity and
produces its own low-rank adaptation of a source-private visual expert.
Image-dependent attention selects and mixes these *features*, not output
predictions. Train with two source indicators, hold out one target indicator.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class DirectedEdgeFusionRegressor(nn.Module):
    def __init__(self, visual_dim, semantic_dim, hidden=128, rank=8, sources=2):
        super().__init__()
        if hidden < 8 or rank < 1 or sources != 2:
            raise ValueError("M2 requires two experts and valid dimensions")
        self.hidden, self.rank, self.sources = hidden, rank, sources
        self.visual = nn.Sequential(
            nn.LayerNorm(visual_dim+4), nn.Linear(visual_dim+4, hidden),
            nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.source_experts = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
            for _ in range(sources)
        ])
        # One relation token per source. Text encoders are frozen upstream.
        self.edge = nn.Sequential(
            nn.LayerNorm(semantic_dim*3+1),
            nn.Linear(semantic_dim*3+1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        # Relation token generates a rank-r adapter and a FiLM mask PER source.
        self.hyper = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden*rank*2 + hidden*2))
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.feature_gate = nn.Linear(hidden, hidden)
        self.target = nn.Sequential(
            nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden),
            nn.GELU())
        self.readout = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.bias = nn.Linear(hidden, 1)
        self.adapter_gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, visual, context, target_semantic, source_semantics,
                edge_relations, self_edges, edge_mask=None, return_attention=False):
        """Shapes: visual [B,D], source_semantics [2,E] or [B,2,E],
        edge_relations [B,2,E], self_edges [B,2].
        edge_mask [B,2] marks allowed paths; at least one must be active.
        """
        batch = len(visual)
        if context.shape != (batch,4) or target_semantic.ndim != 2:
            raise ValueError("Bad visual/context/target shapes")
        if source_semantics.ndim == 2:
            source_semantics = source_semantics.unsqueeze(0).expand(batch,-1,-1)
        if (source_semantics.shape != (batch,2,target_semantic.shape[-1]) or
            edge_relations.shape != source_semantics.shape or
            self_edges.shape != (batch,2)):
            raise ValueError("Bad edge token shapes")
        if edge_mask is None:
            edge_mask = torch.ones(batch,2,dtype=torch.bool,device=visual.device)
        if edge_mask.shape != (batch,2) or not torch.all(edge_mask.any(-1)):
            raise ValueError("Each row needs at least one active source edge")
        encoded = self.visual(torch.cat((visual.float(),context.float()),-1))
        source_h = torch.stack([expert(encoded) for expert in self.source_experts],1)
        e = F.normalize(edge_relations.float(),dim=-1)
        s = F.normalize(source_semantics.float(),dim=-1)
        t = F.normalize(target_semantic.float(),dim=-1)
        edge_inputs = torch.cat([
            s, t.unsqueeze(1).expand(-1,2,-1), e,
            self_edges.float().unsqueeze(-1)], -1)
        tokens = self.edge(edge_inputs)
        generated = self.hyper(tokens)
        block = self.hidden*self.rank
        a, b, gamma, beta = torch.split(
            generated,[block,block,self.hidden,self.hidden],dim=-1)
        a = torch.tanh(a.view(batch,2,self.hidden,self.rank))
        b = torch.tanh(b.view(batch,2,self.rank,self.hidden))
        adapted = torch.matmul(torch.matmul(source_h.unsqueeze(-2),a),b).squeeze(-2)
        adapted = source_h + self.adapter_gain*adapted/self.rank
        # Edge-specific FiLM: relation content influences each feature channel.
        adapted = adapted*(1+0.2*torch.tanh(gamma)) + 0.2*torch.tanh(beta)
        # Visual-dependent attention: query from image+target, keys from each
        # distinct source/edge token and its adapted visual representation.
        target_h = self.target(t)
        query = self.query(encoded+target_h)
        keys = self.key(tokens + adapted)
        logits = (keys*query.unsqueeze(1)).sum(-1)/math.sqrt(self.hidden)
        logits = logits.masked_fill(~edge_mask,-1e9)
        weights = torch.softmax(logits,dim=-1)
        fused = (weights.unsqueeze(-1)*adapted).sum(1)
        out = self.readout(fused) + self.bias(target_h)
        if return_attention:
            return out.squeeze(-1), weights
        return out.squeeze(-1)
