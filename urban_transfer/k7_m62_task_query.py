"""K7 M6.2: indicator-query cross-attention on frozen Qwen-VL visual tokens.

The task query MUST originate from Definition and Relation token groups.
A visual-only shared query is not used. The fused representation preserves
an explicit task-query residual for router and channel gates. MLLM/RemoteCLIP
are frozen; only this small student is trainable.
"""
import torch
import torch.nn.functional as F
from torch import nn


class TaskQueriedRouter(nn.Module):
    def __init__(self, visual_dim, mllm_dim, hidden=128, heads=4,
                 group_counts=(32,16,16)):
        super().__init__()
        counts=tuple(int(x) for x in group_counts)
        if len(counts)!=3 or min(counts)<1 or hidden<8 or hidden%heads:
            raise ValueError("Invalid token group counts or attention hidden size")
        self.group_counts=counts
        self.visual=nn.Sequential(
            nn.LayerNorm(visual_dim+4),nn.Linear(visual_dim+4,hidden),
            nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.image_tokens=nn.Sequential(
            nn.LayerNorm(mllm_dim),nn.Linear(mllm_dim,hidden))
        self.definition_tokens=nn.Sequential(
            nn.LayerNorm(mllm_dim),nn.Linear(mllm_dim,hidden))
        self.relation_tokens=nn.Sequential(
            nn.LayerNorm(mllm_dim),nn.Linear(mllm_dim,hidden))
        self.task_query=nn.Sequential(
            nn.LayerNorm(hidden*2),nn.Linear(hidden*2,hidden),
            nn.GELU(),nn.Linear(hidden,hidden))
        self.cross_attention=nn.MultiheadAttention(
            hidden,heads,batch_first=True)
        # Explicit task-query residual prevents a generic attention pool
        # from being the only input to the router.
        self.fusion=nn.Sequential(
            nn.LayerNorm(hidden*3),nn.Linear(hidden*3,hidden),
            nn.GELU(),nn.LayerNorm(hidden))
        self.router=nn.Sequential(
            nn.Linear(hidden*2,hidden),nn.GELU(),nn.Linear(hidden,2))
        self.gate=nn.Sequential(nn.Linear(hidden*2,hidden),nn.Sigmoid())
        self.experts=nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),
                          nn.GELU(),nn.Linear(hidden,hidden))
            for _ in range(2)])
        self.out=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,1))
        # Source task identity must remain distinguishable given the same
        # visual tile; classifier has NO Education target class or labels.
        self.source_task_head=nn.Linear(hidden,2)
        # Small source-image-only target-branch alignment; NEVER uses
        # Education pixels, labels or city holdout samples in training.
        self.target_align_head=nn.Linear(hidden,visual_dim)

    def split(self,tokens):
        if tokens.ndim!=3 or tokens.shape[1]!=sum(self.group_counts):
            raise ValueError("Expected [B,sum(group_counts),MLLM_dim]")
        return tokens.split(self.group_counts,dim=1)

    def condition(self,tokens):
        visual,definition,relation=self.split(tokens.float())
        d=self.definition_tokens(definition).mean(dim=1)
        r=self.relation_tokens(relation).mean(dim=1)
        task=self.task_query(torch.cat([d,r],dim=-1))
        visual=self.image_tokens(visual)
        attended,_=self.cross_attention(
            task.unsqueeze(1),visual,visual,need_weights=False)
        attended=attended.squeeze(1)
        fused=self.fusion(torch.cat([task,attended,task*attended],dim=-1))
        # Explicit q is used by route+gates, not just a shared image
        # vector, even if the learned fusion tends to ignore task info.
        conditioning=torch.cat([task,fused],dim=-1)
        return conditioning,task,fused

    def forward(self,visual,context,tokens,mask=None,return_aux=False):
        if context.shape!=(len(visual),4) or len(tokens)!=len(visual):
            raise ValueError("Inconsistent RemoteCLIP/context/MLLM batch")
        cond,task,fused=self.condition(tokens)
        h=self.visual(torch.cat([visual.float(),context.float()],dim=-1))
        expert=torch.stack([part(h) for part in self.experts],dim=1)
        expert=expert*self.gate(cond).unsqueeze(1)
        logits=self.router(cond)
        if mask is not None:
            if mask.shape!=logits.shape or not bool(mask.any(-1).all()):
                raise ValueError("Invalid source expert mask")
            logits=logits.masked_fill(~mask,-1e9)
        weights=torch.softmax(logits,dim=-1)
        prediction=self.out((expert*weights.unsqueeze(-1)).sum(dim=1)).squeeze(-1)
        return (prediction,weights,task,fused) if return_aux else prediction

    def source_expert_predictions(self,visual,context,tokens):
        values=[]
        for index in range(2):
            mask=torch.zeros(len(visual),2,device=visual.device,dtype=torch.bool)
            mask[:,index]=True
            values.append(self(visual,context,tokens,mask=mask))
        return torch.stack(values,dim=1)

    def source_task_separation(self,pop_tokens,food_tokens):
        if pop_tokens.shape!=food_tokens.shape:
            raise ValueError("Source paired tokens must share image and shape")
        pop=self.condition(pop_tokens)[1]
        food=self.condition(food_tokens)[1]
        n=len(pop)
        scores=torch.cat([self.source_task_head(pop),
                          self.source_task_head(food)],dim=0)
        target=torch.cat([
            torch.zeros(n,dtype=torch.long,device=pop.device),
            torch.ones(n,dtype=torch.long,device=pop.device)])
        return F.cross_entropy(scores,target)

    def target_image_alignment(self,education_tokens,remoteclip):
        # In-batch positive pairs are allowed to repeat across tasks.
        # We use a pairwise positive cosine objective, not incorrectly
        # labelling duplicate tiles as contrastive negatives.
        knowledge=self.condition(education_tokens)[2]
        estimate=F.normalize(self.target_align_head(knowledge),dim=-1)
        actual=F.normalize(remoteclip.float().detach(),dim=-1)
        return (1-(estimate*actual).sum(dim=-1)).mean()
