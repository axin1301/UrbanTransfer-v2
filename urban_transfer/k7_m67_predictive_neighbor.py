"""M6.7 predictive cross-indicator neighbor context.

Same architecture and parameter count for cross/same/zero arms:
- cross_indicator: neighboring predicted source A and other B
- same_indicator: two copies of neighboring predicted source A
- no_predictive: no neighboring predicted indicators
All keep identical M6.5 real unlabeled visual neighborhood inputs.
"""
import torch
import torch.nn.functional as F
from torch import nn

from urban_transfer.k7_m65_neighborhood_student import (
    NeighborhoodCityIndicatorResidual)


class PredictiveNeighborResidual(NeighborhoodCityIndicatorResidual):
    def __init__(self,visual_dim,mllm_dim,*,hidden=128,heads=4,
                 group_counts=(32,16,16),max_residual=.5):
        super().__init__(visual_dim,mllm_dim,hidden=hidden,heads=heads,
                         group_counts=group_counts,max_residual=max_residual)
        self.predicted_encoder=nn.Sequential(
            nn.LayerNorm(4),nn.Linear(4,hidden),nn.GELU(),
            nn.Linear(hidden,hidden))
        self.predictive_attention=nn.MultiheadAttention(
            hidden,heads,batch_first=True)
        self.predictive_query=nn.Sequential(
            nn.LayerNorm(2*hidden),nn.Linear(2*hidden,hidden),
            nn.GELU(),nn.LayerNorm(hidden))
        self.predictive_fusion=nn.Sequential(
            nn.LayerNorm(2*hidden),nn.Linear(2*hidden,hidden),
            nn.GELU(),nn.LayerNorm(hidden))

    def condition(self,visual,context,tokens,neighborhood,
                  neighbor_predictions,neighbor_distance,neighbor_mask):
        joint,city,task,fused,attended=super().condition(
            visual,context,tokens,neighborhood)
        batch=len(visual)
        if (neighbor_predictions.ndim!=3 or
            neighbor_predictions.shape[:1]!=(batch,) or
            neighbor_predictions.shape[-1]!=2 or
            neighbor_distance.shape!=neighbor_predictions.shape[:2] or
            neighbor_mask.shape!=neighbor_distance.shape or
            neighbor_mask.dtype!=torch.bool or
            not bool(neighbor_mask.any(dim=-1).all())):
            raise ValueError("M6.7 expects predictions[B,K,2], distance[B,K], valid neighbor mask")
        values=neighbor_predictions.float()
        d=neighbor_distance.float().unsqueeze(-1)
        features=torch.cat((
            values,values[...,0:1]*values[...,1:2],d),dim=-1)
        keys=self.predicted_encoder(features)
        query=self.predictive_query(
            torch.cat((city,task),dim=-1)).unsqueeze(1)
        attended_values,_=self.predictive_attention(
            query,keys,keys,key_padding_mask=~neighbor_mask,
            need_weights=False)
        new_fused=self.predictive_fusion(torch.cat(
            (fused,attended_values.squeeze(1)),dim=-1))
        new_joint=torch.cat((city,task,new_fused),dim=-1)
        return new_joint,city,task,new_fused,attended

    def forward(self,visual,context,tokens,neighborhood,
                neighbor_predictions,neighbor_distance,neighbor_mask,
                mask=None,return_aux=False):
        base=self.base(visual,context)
        joint,city,task,fused,_=self.condition(
            visual,context,tokens,neighborhood,
            neighbor_predictions,neighbor_distance,neighbor_mask)
        logits=self.router(joint)
        if mask is not None:
            if mask.shape!=logits.shape or not bool(mask.any(-1).all()):
                raise ValueError("Invalid M6.7 expert mask")
            logits=logits.masked_fill(~mask,-1e9)
        weights=F.softmax(logits,dim=-1)
        expert_input=torch.cat((city,fused),dim=-1)
        residuals=torch.cat(
            [expert(expert_input) for expert in self.residual_experts],dim=1)
        raw=(weights*residuals).sum(-1,keepdim=True)
        gate=torch.sigmoid(self.residual_gate(joint))
        delta=self.max_residual*torch.tanh(
            self.residual_scale(torch.tanh(raw)*gate)).squeeze(-1)
        prediction=base+delta
        return ((prediction,weights,task,fused,base,delta)
                if return_aux else prediction)

    def source_expert_predictions(self,visual,context,tokens,neighborhood,
                                  npred,ndist,nmask):
        results=[]
        for idx in range(2):
            mask=torch.zeros((len(visual),2),dtype=torch.bool,
                             device=visual.device)
            mask[:,idx]=True
            results.append(self(visual,context,tokens,neighborhood,
                                npred,ndist,nmask,mask=mask))
        return torch.stack(results,dim=1)

    def source_task_separation(self,visual,context,pop_tokens,food_tokens,
                               neighborhood,npred,ndist,nmask):
        pop=self.condition(visual,context,pop_tokens,neighborhood,
                           npred,ndist,nmask)[2]
        food=self.condition(visual,context,food_tokens,neighborhood,
                            npred,ndist,nmask)[2]
        labels=torch.cat((
            torch.zeros(len(pop),device=visual.device,dtype=torch.long),
            torch.ones(len(food),device=visual.device,dtype=torch.long)))
        return F.cross_entropy(
            torch.cat((self.source_task_head(pop),
                       self.source_task_head(food)),dim=0),labels)

    def target_image_alignment(self,visual,context,edu_tokens,
                               neighborhood,npred,ndist,nmask):
        fused=self.condition(visual,context,edu_tokens,
                             neighborhood,npred,ndist,nmask)[3]
        projected=F.normalize(self.target_align_head(fused),dim=-1)
        reference=F.normalize(visual.float().detach(),dim=-1)
        return (1-(projected*reference).sum(-1)).mean()
