"""M6.5 neighborhood-aware City×Indicator residual student.

Same source-only losses, MLLM token grouping and 0.5-standardized
bounded transfer concept as M6.3. Additional unlabeled physical-tile
neighborhood mean, dispersion and contrast condition spatial queries.
"""
import torch
import torch.nn.functional as F
from torch import nn
from urban_transfer.k7_m63_city_indicator_residual import CityIndicatorResidual


class NeighborhoodCityIndicatorResidual(CityIndicatorResidual):
    def __init__(self,visual_dim,mllm_dim,*,hidden=128,heads=4,
                 group_counts=(32,16,16),max_residual=.5):
        super().__init__(visual_dim,mllm_dim,hidden=hidden,
                         heads=heads,group_counts=group_counts,
                         max_residual=max_residual)
        self.neighborhood_dim=3*visual_dim+2
        self.neighborhood_encoder=nn.Sequential(
            nn.LayerNorm(self.neighborhood_dim),
            nn.Linear(self.neighborhood_dim,hidden),nn.GELU(),
            nn.Linear(hidden,hidden))
        self.neighborhood_gate=nn.Sequential(
            nn.Linear(hidden*2,hidden),nn.Sigmoid())
        self.neighborhood_norm=nn.LayerNorm(hidden)

    def condition(self,visual,context,tokens,neighborhood=None):
        if (visual.ndim!=2 or context.shape!=(len(visual),4)
            or tokens.ndim!=3 or tokens.shape[0]!=len(visual)
            or neighborhood is None
            or neighborhood.shape!=(len(visual),self.neighborhood_dim)):
            raise ValueError("M6.5 expects visual, [B,4] geo, cached tokens and [B,3*D+2] neighborhood")
        image,definition,relation=self.split(tokens)
        local=self.spatial(torch.cat([visual.float(),context.float()],dim=-1))
        nearby=self.neighborhood_encoder(neighborhood.float())
        gate=self.neighborhood_gate(torch.cat([local,nearby],dim=-1))
        city=self.neighborhood_norm(local+gate*nearby)
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

    def forward(self,visual,context,tokens,neighborhood,
                mask=None,return_aux=False):
        base=self.base(visual,context)
        joint,city,task,fused,_=self.condition(
            visual,context,tokens,neighborhood)
        logits=self.router(joint)
        if mask is not None:
            if mask.shape!=logits.shape or not bool(mask.any(-1).all()):
                raise ValueError("Invalid M6.5 expert mask")
            logits=logits.masked_fill(~mask,-1e9)
        weights=F.softmax(logits,dim=-1)
        expert_input=torch.cat([city,fused],dim=-1)
        residuals=torch.cat([e(expert_input)
                             for e in self.residual_experts],dim=1)
        raw=(weights*residuals).sum(dim=-1,keepdim=True)
        gate=torch.sigmoid(self.residual_gate(joint))
        delta=self.max_residual*torch.tanh(
            self.residual_scale(torch.tanh(raw)*gate)).squeeze(-1)
        prediction=base+delta
        return ((prediction,weights,task,fused,base,delta)
                if return_aux else prediction)

    def source_expert_predictions(self,visual,context,tokens,neighborhood):
        result=[]
        for index in range(2):
            mask=torch.zeros(len(visual),2,dtype=torch.bool,device=visual.device)
            mask[:,index]=True
            result.append(self(visual,context,tokens,neighborhood,mask=mask))
        return torch.stack(result,dim=1)

    def source_task_separation(self,visual,context,pop_tokens,food_tokens,
                               neighborhood):
        pop=self.condition(visual,context,pop_tokens,neighborhood)[2]
        food=self.condition(visual,context,food_tokens,neighborhood)[2]
        labels=torch.cat([
            torch.zeros(len(pop),device=visual.device,dtype=torch.long),
            torch.ones(len(food),device=visual.device,dtype=torch.long)])
        return F.cross_entropy(
            torch.cat([self.source_task_head(pop),
                       self.source_task_head(food)],dim=0),labels)

    def target_image_alignment(self,visual,context,education_tokens,
                               neighborhood):
        fused=self.condition(visual,context,education_tokens,neighborhood)[3]
        projected=F.normalize(self.target_align_head(fused),dim=-1)
        target=F.normalize(visual.float().detach(),dim=-1)
        return (1-(projected*target).sum(-1)).mean()
