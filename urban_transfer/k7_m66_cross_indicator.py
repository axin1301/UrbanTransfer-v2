"""M6.6 neighbor × indicator × target interaction with source-only prompts.

The model sees explicit K neighbor visual embeddings and fixed, label-free
Qwen Definition+Relation prototypes for Population/Food/Education. A
target-indicator Query attends neighbor × indicator combinations.

Modes have the SAME parameter count:
  cross_indicator: neighbor × {Pop,Food,Education} semantic prototypes;
  spatial_only: neighbor × {shared shared shared} pooled semantic baseline.
Both still include the original M6.5 aggregate-neighborhood vector,
and both use identical visual-only anchor.
"""
import torch
import torch.nn.functional as F
from torch import nn

from urban_transfer.k7_m65_neighborhood_student import (
    NeighborhoodCityIndicatorResidual)


class NeighborIndicatorInteraction(NeighborhoodCityIndicatorResidual):
    def __init__(self,visual_dim,mllm_dim,semantic_prototypes,*,mode,
                 hidden=128,heads=4,group_counts=(32,16,16),
                 max_residual=.5):
        super().__init__(visual_dim,mllm_dim,hidden=hidden,heads=heads,
                         group_counts=group_counts,max_residual=max_residual)
        if mode not in ("cross_indicator","spatial_only"):
            raise ValueError("Unsupported M6.6 comparison arm")
        if tuple(semantic_prototypes.shape)!=(3,2*mllm_dim):
            raise ValueError("Expected 3 source-image prompt prototypes [3,2*mllm_dim]")
        self.mode=mode
        self.register_buffer("semantic_prototypes",
                             semantic_prototypes.detach().float().clone())
        self.neighbor_visual_project=nn.Sequential(
            nn.LayerNorm(visual_dim),nn.Linear(visual_dim,hidden),
            nn.GELU(),nn.Linear(hidden,hidden))
        self.neighbor_semantic_project=nn.Sequential(
            nn.LayerNorm(2*mllm_dim),nn.Linear(2*mllm_dim,hidden),
            nn.GELU(),nn.Linear(hidden,hidden))
        self.distance_project=nn.Sequential(
            nn.Linear(1,hidden),nn.Tanh())
        self.cross_token=nn.Sequential(
            nn.LayerNorm(4*hidden),nn.Linear(4*hidden,hidden),
            nn.GELU(),nn.LayerNorm(hidden))
        self.cross_query=nn.Sequential(
            nn.LayerNorm(3*hidden),nn.Linear(3*hidden,hidden),
            nn.GELU(),nn.LayerNorm(hidden))
        self.neighbor_task_attention=nn.MultiheadAttention(
            hidden,heads,batch_first=True)
        self.cross_fusion=nn.Sequential(
            nn.LayerNorm(2*hidden),nn.Linear(2*hidden,hidden),
            nn.GELU(),nn.LayerNorm(hidden))

    def condition(self,visual,context,tokens,neighborhood,
                  neighbor_visual,neighbor_distance,neighbor_mask):
        joint,city,task,fused,attended=super().condition(
            visual,context,tokens,neighborhood)
        b=len(visual)
        if (neighbor_visual.ndim!=3 or neighbor_visual.shape[0]!=b or
            neighbor_visual.shape[2]!=visual.shape[1] or
            neighbor_distance.shape!=neighbor_visual.shape[:2] or
            neighbor_mask.shape!=neighbor_distance.shape or
            neighbor_mask.dtype!=torch.bool):
            raise ValueError("M6.6 expects neighbor_visual[B,K,D], distance[B,K], bool mask[B,K]")
        if not bool(neighbor_mask.any(-1).all()):
            raise ValueError("Each M6.6 row requires at least one physical neighbor")
        k=neighbor_visual.shape[1]
        h=city.shape[-1]
        image=self.neighbor_visual_project(neighbor_visual.float())
        projected=self.neighbor_semantic_project(self.semantic_prototypes)
        if self.mode=="cross_indicator":
            # The three B identities remain distinct at each neighbor.
            semantics=projected.view(1,1,3,h).expand(b,k,3,h)
        else:
            # Identical source prompt prototypes but collapse B identity.
            # Same parameterized encoder and number of attention slots,
            # same target-A query, same neighbors and neighborhood vector.
            pooled=projected.mean(0).view(1,1,1,h)
            semantics=pooled.expand(b,k,3,h)
        v=image.unsqueeze(2).expand(b,k,3,h)
        dist=self.distance_project(
            neighbor_distance.float().unsqueeze(-1))
        dist=dist.unsqueeze(2).expand(b,k,3,h)
        raw=self.cross_token(torch.cat((v,semantics,v*semantics,dist),dim=-1))
        keys=raw.reshape(b,k*3,h)
        pad=(~neighbor_mask).unsqueeze(-1).expand(b,k,3).reshape(b,k*3)
        query=self.cross_query(torch.cat(
            (city,task,city*task),dim=-1)).unsqueeze(1)
        selected,_=self.neighbor_task_attention(
            query,keys,keys,key_padding_mask=pad,need_weights=False)
        cross=selected.squeeze(1)
        fused=self.cross_fusion(torch.cat((fused,cross),dim=-1))
        return torch.cat((city,task,fused),dim=-1),city,task,fused,attended

    def forward(self,visual,context,tokens,neighborhood,
                neighbor_visual,neighbor_distance,neighbor_mask,
                mask=None,return_aux=False):
        base=self.base(visual,context)
        joint,city,task,fused,_=self.condition(
            visual,context,tokens,neighborhood,
            neighbor_visual,neighbor_distance,neighbor_mask)
        logits=self.router(joint)
        if mask is not None:
            if mask.shape!=logits.shape or not bool(mask.any(-1).all()):
                raise ValueError("Invalid M6.6 expert mask")
            logits=logits.masked_fill(~mask,-1e9)
        weights=F.softmax(logits,dim=-1)
        expert_input=torch.cat((city,fused),dim=-1)
        residuals=torch.cat(
            [expert(expert_input) for expert in self.residual_experts],dim=1)
        raw=(weights*residuals).sum(-1,keepdim=True)
        gate=torch.sigmoid(self.residual_gate(joint))
        delta=self.max_residual*torch.tanh(
            self.residual_scale(torch.tanh(raw)*gate)).squeeze(-1)
        pred=base+delta
        return ((pred,weights,task,fused,base,delta)
                if return_aux else pred)

    def source_expert_predictions(self,visual,context,tokens,
                                  neighborhood,nv,nd,nm):
        preds=[]
        for i in range(2):
            mask=torch.zeros((len(visual),2),dtype=torch.bool,
                             device=visual.device)
            mask[:,i]=True
            preds.append(self(visual,context,tokens,neighborhood,
                              nv,nd,nm,mask=mask))
        return torch.stack(preds,dim=1)

    def source_task_separation(self,visual,context,pop_tokens,food_tokens,
                               neighborhood,nv,nd,nm):
        pop=self.condition(visual,context,pop_tokens,neighborhood,
                           nv,nd,nm)[2]
        food=self.condition(visual,context,food_tokens,neighborhood,
                            nv,nd,nm)[2]
        labels=torch.cat((
            torch.zeros(len(pop),device=visual.device,dtype=torch.long),
            torch.ones(len(food),device=visual.device,dtype=torch.long)))
        return F.cross_entropy(
            torch.cat((self.source_task_head(pop),
                       self.source_task_head(food)),dim=0),labels)

    def target_image_alignment(self,visual,context,education_tokens,
                               neighborhood,nv,nd,nm):
        fused=self.condition(visual,context,education_tokens,
                             neighborhood,nv,nd,nm)[3]
        pred=F.normalize(self.target_align_head(fused),dim=-1)
        truth=F.normalize(visual.float().detach(),dim=-1)
        return (1-(pred*truth).sum(-1)).mean()
