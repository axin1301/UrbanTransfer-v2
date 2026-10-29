"""New architecture. Experts are learned latent functions, NOT named continent/income experts."""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    visual_dim: int = 768
    context_dim: int = 4
    task_dim: int = 128
    hidden_dim: int = 128
    n_experts: int = 6
    top_k: int = 2
    n_sources: int = 5
    use_knowledge: bool = True

class TransferModel(nn.Module):
    """Continuous regression from image features, geography, text, and online knowledge."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.top_k < 1 or cfg.top_k > cfg.n_experts: raise ValueError('invalid top_k')
        self.cfg = cfg
        d=cfg.hidden_dim
        self.visual=nn.Sequential(nn.Linear(cfg.visual_dim,d),nn.LayerNorm(d),nn.GELU())
        self.context=nn.Sequential(nn.Linear(cfg.context_dim,d),nn.LayerNorm(d),nn.GELU())
        self.task=nn.Sequential(nn.Linear(cfg.task_dim,d),nn.LayerNorm(d),nn.GELU())
        self.knowledge=nn.Sequential(nn.Linear(cfg.n_sources+4,d),nn.GELU(),nn.Linear(d,d))
        self.base_router=nn.Linear(3*d,cfg.n_experts)
        self.knowledge_router=nn.Linear(d,cfg.n_experts,bias=False)
        nn.init.zeros_(self.knowledge_router.weight)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,d)) for _ in range(cfg.n_experts)])
        self.shared=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,d))
        self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,1))
    def forward(self, visual, context, task, knowledge=None, return_aux=False):
        b=visual.shape[0]
        v=self.visual(visual); c=self.context(context); t=self.task(task)
        h=v+c+t
        logits=self.base_router(torch.cat([v,c,t],dim=-1))
        if self.cfg.use_knowledge:
            if knowledge is None: raise ValueError('Knowledge required when use_knowledge=True')
            if knowledge.shape != (b,self.cfg.n_sources+4): raise ValueError('Invalid knowledge shape')
            logits=logits+self.knowledge_router(self.knowledge(knowledge))
        vals,ids=logits.topk(self.cfg.top_k,dim=-1)
        weights=F.softmax(vals,dim=-1)
        outputs=torch.stack([expert(h) for expert in self.experts],dim=1)
        selected=outputs.gather(1,ids.unsqueeze(-1).expand(-1,-1,h.shape[-1]))
        combined=(selected*weights.unsqueeze(-1)).sum(dim=1)
        pred=self.head(h+self.shared(h)+combined).squeeze(-1)
        if return_aux:
            return pred,{'expert_indices':ids,'expert_weights':weights,'router_logits':logits}
        return pred
