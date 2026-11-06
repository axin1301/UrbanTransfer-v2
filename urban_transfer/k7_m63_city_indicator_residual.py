"""M6.3 City×Indicator Interaction with bounded visual residual transfer.

A pre-trained frozen RemoteCLIP+geo two-expert Visual-only student is
the prediction anchor. Its source-only checkpoint is NEVER finetuned.

Task query comes from genuine Qwen-VL Definition+Relation hidden tokens.
The same Qwen-VL visual tokens are centered (per image) and queried by
a City×Task query driven by RemoteCLIP+geography and task semantics.
The residual is bounded in source-standardized output units and starts
EXACTLY at zero. Only source Population/Food labels supervise fitting.
"""
import torch
import torch.nn.functional as F
from torch import nn

from scripts.run_k7_m62_fixed_comparison import TwoExpertControl


class CityIndicatorResidual(nn.Module):
    def __init__(self,visual_dim,mllm_dim,*,hidden=128,heads=4,
                 group_counts=(32,16,16),max_residual=.5):
        super().__init__()
        counts=tuple(int(v) for v in group_counts)
        if (len(counts)!=3 or min(counts)<2 or hidden<8 or
            hidden%heads or not 0<max_residual<=1.):
            raise ValueError("Invalid M6.3 architecture or residual bound")
        self.group_counts=counts
        self.max_residual=float(max_residual)
        self.base=TwoExpertControl(visual_dim,text_dim=0,hidden=hidden)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.spatial=nn.Sequential(
            nn.LayerNorm(visual_dim+4),
            nn.Linear(visual_dim+4,hidden),nn.GELU(),
            nn.Linear(hidden,hidden),nn.LayerNorm(hidden))
        self.visual_tokens=nn.Sequential(
            nn.LayerNorm(mllm_dim),nn.Linear(mllm_dim,hidden))
        self.definition_tokens=nn.Sequential(
            nn.LayerNorm(mllm_dim),nn.Linear(mllm_dim,hidden))
        self.relation_tokens=nn.Sequential(
            nn.LayerNorm(mllm_dim),nn.Linear(mllm_dim,hidden))
        self.task_query=nn.Sequential(
            nn.LayerNorm(hidden*2),nn.Linear(hidden*2,hidden),
            nn.GELU(),nn.Linear(hidden,hidden))
        self.city_task_query=nn.Sequential(
            nn.LayerNorm(hidden*3),nn.Linear(hidden*3,hidden),
            nn.GELU(),nn.LayerNorm(hidden))
        self.cross_attention=nn.MultiheadAttention(
            hidden,heads,batch_first=True)
        # Centering suppresses a common content offset, but the model
        # retains the uncentered visual mean through 'global_visual'.
        self.global_visual=nn.Sequential(
            nn.LayerNorm(hidden),nn.Linear(hidden,hidden))
        self.fusion=nn.Sequential(
            nn.LayerNorm(hidden*4),nn.Linear(hidden*4,hidden),
            nn.GELU(),nn.LayerNorm(hidden))
        self.router=nn.Sequential(
            nn.LayerNorm(hidden*3),nn.Linear(hidden*3,hidden),
            nn.GELU(),nn.Linear(hidden,2))
        self.residual_experts=nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden*2),
                          nn.Linear(hidden*2,hidden),
                          nn.GELU(),nn.Linear(hidden,1))
            for _ in range(2)])
        self.residual_gate=nn.Sequential(
            nn.LayerNorm(hidden*3),nn.Linear(hidden*3,hidden),
            nn.GELU(),nn.Linear(hidden,1))
        # Zero initializes the transfer branch while retaining gradients
        # to gate/expert networks after the first optimization step.
        self.residual_scale=nn.Linear(1,1,bias=False)
        nn.init.zeros_(self.residual_scale.weight)
        self.source_task_head=nn.Linear(hidden,2)
        self.target_align_head=nn.Linear(hidden,visual_dim)

    def load_visual_baseline(self,state_dict):
        self.base.load_state_dict(state_dict,strict=True)
        for param in self.base.parameters():
            param.requires_grad_(False)

    def split(self,tokens):
        if tokens.ndim!=3 or tokens.shape[1]!=sum(self.group_counts):
            raise ValueError("Expected [B,sum(group_counts),MLLM_dim]")
        return tokens.float().split(self.group_counts,dim=1)

    def condition(self,visual,context,tokens):
        if (visual.ndim!=2 or context.shape!=(len(visual),4) or
            tokens.shape[0]!=len(visual)):
            raise ValueError("Inconsistent City/Indicator/RemoteCLIP batch")
        image,definition,relation=self.split(tokens)
        city=self.spatial(torch.cat([visual.float(),context.float()],dim=-1))
        d=self.definition_tokens(definition).mean(1)
        r=self.relation_tokens(relation).mean(1)
        task=self.task_query(torch.cat([d,r],dim=-1))
        query=self.city_task_query(torch.cat([city,task,city*task],dim=-1))
        visual_keys=self.visual_tokens(image)
        global_visual=visual_keys.mean(1)
        centered=visual_keys-global_visual.unsqueeze(1)
        attended,_=self.cross_attention(
            query.unsqueeze(1),centered,centered,need_weights=False)
        attended=attended.squeeze(1)
        fused=self.fusion(torch.cat(
            [city,task,attended,self.global_visual(global_visual)],dim=-1))
        joint=torch.cat([city,task,fused],dim=-1)
        return joint,city,task,fused,attended

    def forward(self,visual,context,tokens,mask=None,return_aux=False):
        # Validate BEFORE invoking the frozen visual anchor: the anchor's
        # LayerNorm otherwise raises an opaque RuntimeError for malformed
        # context (e.g. 3 geo fields instead of the required 4).
        if (visual.ndim!=2 or context.shape!=(len(visual),4) or
            tokens.ndim!=3 or tokens.shape[0]!=len(visual) or
            tokens.shape[1]!=sum(self.group_counts)):
            raise ValueError(
                "M6.3 expects visual=[B,D], context=[B,4], "
                "tokens=[B,sum(group_counts),MLLM_dim]")
        base=self.base(visual,context)
        joint,city,task,fused,attended=self.condition(
            visual,context,tokens)
        logits=self.router(joint)
        if mask is not None:
            if mask.shape!=logits.shape or not bool(mask.any(-1).all()):
                raise ValueError("Invalid residual expert mask")
            logits=logits.masked_fill(~mask,-1e9)
        weights=F.softmax(logits,dim=-1)
        expert_input=torch.cat([city,fused],dim=-1)
        residuals=torch.cat(
            [expert(expert_input) for expert in self.residual_experts],dim=1)
        raw=(weights*residuals).sum(dim=-1,keepdim=True)
        gate=torch.sigmoid(self.residual_gate(joint))
        # double tanh ensures strict |delta| <= max_residual. Effective
        # residual gate is image+indicator dependent, not task static.
        transfer=self.max_residual*torch.tanh(
            self.residual_scale(torch.tanh(raw)*gate)).squeeze(-1)
        prediction=base+transfer
        if return_aux:
            return prediction,weights,task,fused,base,transfer
        return prediction

    def source_expert_predictions(self,visual,context,tokens):
        result=[]
        for k in range(2):
            mask=torch.zeros(len(visual),2,dtype=torch.bool,
                             device=visual.device)
            mask[:,k]=True
            result.append(self(visual,context,tokens,mask))
        return torch.stack(result,dim=1)

    def source_task_separation(self,visual,context,pop_tokens,food_tokens):
        if pop_tokens.shape!=food_tokens.shape:
            raise ValueError("Same-image source tokens required")
        pop=self.condition(visual,context,pop_tokens)[2]
        food=self.condition(visual,context,food_tokens)[2]
        label=torch.cat([
            torch.zeros(len(pop),device=visual.device,dtype=torch.long),
            torch.ones(len(food),device=visual.device,dtype=torch.long)])
        return F.cross_entropy(
            torch.cat([self.source_task_head(pop),
                       self.source_task_head(food)],dim=0),label)

    def target_image_alignment(self,visual,context,education_tokens):
        fused=self.condition(visual,context,education_tokens)[3]
        projected=F.normalize(self.target_align_head(fused),dim=-1)
        target=F.normalize(visual.float().detach(),dim=-1)
        return (1-(projected*target).sum(-1)).mean()
