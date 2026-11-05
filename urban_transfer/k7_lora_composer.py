"""K7 M3 Lite: source-specific LoRA matrices, rank-wise directed relation gates."""
import math
import torch
import torch.nn.functional as F
from torch import nn

class RelationLoRAComposer(nn.Module):
    """Delta W = sum_s alpha_s B_s diag(g_s) A_s, no text averaging."""
    def __init__(self, visual_dim, semantic_dim, hidden=128, rank=8):
        super().__init__()
        if hidden < 8 or rank < 1:
            raise ValueError("Invalid dimensions")
        self.rank = rank
        self.trunk = nn.Sequential(nn.LayerNorm(visual_dim+4),
            nn.Linear(visual_dim+4,hidden),nn.GELU(),
            nn.Linear(hidden,hidden),nn.GELU())
        self.source_a = nn.Parameter(torch.empty(2,rank,hidden))
        self.source_b = nn.Parameter(torch.zeros(2,hidden,rank))
        for a in self.source_a:
            nn.init.kaiming_uniform_(a,a=math.sqrt(5))
        self.task = nn.Sequential(nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim,hidden),nn.GELU())
        self.edge = nn.Sequential(nn.LayerNorm(semantic_dim*3+1),
            nn.Linear(semantic_dim*3+1,hidden),nn.GELU(),
            nn.Linear(hidden,hidden),nn.GELU())
        self.rank_gate = nn.Linear(hidden,rank)
        self.router = nn.Linear(hidden,1)
        self.key = nn.Linear(hidden,hidden)
        self.query = nn.Linear(hidden*2,hidden)
        self.readout = nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,1))
        self.bias = nn.Linear(hidden,1)
        self.gain = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.rank_gate.weight)
        nn.init.zeros_(self.rank_gate.bias)

    def forward(self,visual,context,target_semantic,source_semantics,
                edge_relations,self_edges,edge_mask=None,return_aux=False):
        n = len(visual)
        if context.shape != (n,4) or target_semantic.ndim != 2 or len(target_semantic)!=n:
            raise ValueError("Invalid visual, context or target shapes")
        if source_semantics.ndim == 2:
            source_semantics = source_semantics[None].expand(n,-1,-1)
        shape = (n,2,target_semantic.shape[-1])
        if source_semantics.shape != shape or edge_relations.shape != shape or self_edges.shape != (n,2):
            raise ValueError("Expected two directed source edge tokens")
        if edge_mask is None:
            edge_mask = torch.ones((n,2),device=visual.device,dtype=torch.bool)
        if edge_mask.shape != (n,2) or not bool(edge_mask.any(-1).all()):
            raise ValueError("Every sample needs at least one active source")
        h = self.trunk(torch.cat((visual.float(),context.float()),-1))
        t = F.normalize(target_semantic.float(),dim=-1)
        s = F.normalize(source_semantics.float(),dim=-1)
        r = F.normalize(edge_relations.float(),dim=-1)
        edge = self.edge(torch.cat((
            s,t[:,None,:].expand(-1,2,-1),r,self_edges.float().unsqueeze(-1)),-1))
        target_h = self.task(t)
        gates = 2.0*torch.sigmoid(self.rank_gate(edge))
        low = torch.einsum("nh,srh->nsr",h,self.source_a)*gates
        updates = torch.einsum("nsr,shr->nsh",low,self.source_b)/self.rank
        query = self.query(torch.cat((h,target_h),-1))
        logits = self.router(edge).squeeze(-1) + (self.key(edge)*query[:,None,:]).sum(-1)/math.sqrt(query.shape[-1])
        mix = torch.softmax(logits.masked_fill(~edge_mask,-1e9),dim=-1)
        adapted = h+self.gain*(updates*mix.unsqueeze(-1)).sum(1)
        pred = (self.readout(adapted)+self.bias(target_h)).squeeze(-1)
        return (pred,mix,gates) if return_aux else pred
