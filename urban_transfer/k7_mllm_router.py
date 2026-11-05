"""K7 M6: task-conditioned MLLM hidden-token cross-attention router.

Frozen Qwen-VL supplies per-image/per-task hidden states. The student
receives RemoteCLIP features and geography, while its *expert routing*
must pass through an MLLM token cross-attention readout. Two source
experts are supervised by their own source indicator labels.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class MLLMHiddenRouter(nn.Module):
    def __init__(self,visual_dim,mllm_dim,hidden=128,heads=4):
        super().__init__()
        if hidden%heads or hidden<8:
            raise ValueError("Hidden must be divisible by attention heads")
        self.visual=nn.Sequential(
            nn.LayerNorm(visual_dim+4),nn.Linear(visual_dim+4,hidden),
            nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.mllm_tokens=nn.Sequential(
            nn.LayerNorm(mllm_dim),nn.Linear(mllm_dim,hidden),
            nn.GELU(),nn.LayerNorm(hidden))
        self.query=nn.Linear(hidden,hidden)
        self.attn=nn.MultiheadAttention(hidden,heads,batch_first=True)
        self.cross_gate=nn.Sequential(
            nn.Linear(hidden,hidden),nn.Sigmoid())
        self.experts=nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),
                          nn.GELU(),nn.Linear(hidden,hidden))
            for _ in range(2)
        ])
        self.router=nn.Sequential(
            nn.LayerNorm(hidden),nn.Linear(hidden,hidden),
            nn.GELU(),nn.Linear(hidden,2))
        self.readout=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,1))

    def forward(self,visual,context,hidden_tokens,mask=None,return_weights=False):
        n=len(visual)
        if (context.shape!=(n,4) or hidden_tokens.ndim!=3 or
            hidden_tokens.shape[0]!=n):
            raise ValueError("Invalid MLLM router input shapes")
        h=self.visual(torch.cat((visual.float(),context.float()),dim=-1))
        tokens=self.mllm_tokens(hidden_tokens.float())
        query=self.query(h).unsqueeze(1)
        fused,_=self.attn(query,tokens,tokens,need_weights=False)
        fused=fused.squeeze(1)
        # Neither the routing logits nor task-specific channel gates
        # have a text-free shortcut: both depend on cross-attended MLLM.
        gates=self.cross_gate(fused)
        expert_features=torch.stack([expert(h) for expert in self.experts],dim=1)
        expert_features=expert_features*gates[:,None,:]
        logits=self.router(fused)
        if mask is not None:
            if mask.shape!=(n,2) or not bool(mask.any(-1).all()):
                raise ValueError("Every row needs at least one source expert")
            logits=logits.masked_fill(~mask,-1e9)
        weights=torch.softmax(logits,dim=-1)
        z=(expert_features*weights.unsqueeze(-1)).sum(1)
        prediction=self.readout(z).squeeze(-1)
        return (prediction,weights) if return_weights else prediction


def mllm_teacher_distribution(source_tokens,target_tokens,temperature=.2,
                              floor=.1):
    """Per-image teacher: similarity of 3 task-conditioned MLLM states.

    source_tokens [B,2,L,D], target_tokens [B,L,D]. No labels.
    This is a proxy for relatedness, NOT known source transfer utility.
    """
    if (source_tokens.ndim!=4 or source_tokens.shape[1]!=2 or
        target_tokens.shape!=source_tokens.shape[:1]+source_tokens.shape[2:]
        or temperature<=0 or not 0<=floor<.5):
        raise ValueError("Invalid teacher tokens or temperature")
    a=F.normalize(source_tokens.float().mean(-2),dim=-1)
    b=F.normalize(target_tokens.float().mean(-2),dim=-1)
    scores=(a*b[:,None,:]).sum(-1)
    probs=F.softmax(scores/temperature,dim=-1)
    return ((1-2*floor)*probs+floor).detach()


def router_distillation(weights,teacher):
    if weights.shape!=teacher.shape or weights.ndim!=2:
        raise ValueError("Invalid router KD shape")
    return F.kl_div(weights.clamp_min(1e-8).log(),teacher,
                    reduction="batchmean")
