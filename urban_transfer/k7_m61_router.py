"""M6.1: structured MLLM token-guided two-expert regression.

Cross attention has a learned query, not the RemoteCLIP image feature:
the router cannot bypass the MLLM by copying a visual query.
Visual expert features remain distinct and receive source regression
supervision. Target text enters image-text contrastive training without
target regression labels or proxy source importance probabilities.
"""
import torch
import torch.nn.functional as F
from torch import nn


class StructuredMLLMRouter(nn.Module):
    def __init__(self,visual_dim,mllm_dim,hidden=128,heads=4,
                 group_counts=(32,16,16)):
        super().__init__()
        if hidden<8 or hidden%heads or len(group_counts)!=3 or min(group_counts)<1:
            raise ValueError("Invalid M6.1 hidden dimension/group budgets")
        self.group_counts=tuple(group_counts)
        self.visual=nn.Sequential(
            nn.LayerNorm(visual_dim+4),nn.Linear(visual_dim+4,hidden),
            nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.token_projection=nn.Sequential(
            nn.LayerNorm(mllm_dim),nn.Linear(mllm_dim,hidden))
        self.segment=nn.Embedding(3,hidden)
        self.query=nn.Parameter(torch.randn(1,1,hidden)*.02)
        self.attn=nn.MultiheadAttention(hidden,heads,batch_first=True)
        self.fusion=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),
                                  nn.GELU())
        self.gate=nn.Sequential(nn.Linear(hidden,hidden),nn.Sigmoid())
        self.router=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,2))
        self.experts=nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),
                          nn.GELU(),nn.Linear(hidden,hidden)) for _ in range(2)])
        self.out=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,1))
        # The contrastive target branch maps MLLM-only fusion to frozen
        # RemoteCLIP space. There is no visual-x shortcut in this pathway.
        self.align=nn.Linear(hidden,visual_dim)

    def read_hidden(self,ml_tokens):
        if ml_tokens.ndim!=3 or ml_tokens.shape[1]!=sum(self.group_counts):
            raise ValueError("Expected [B,total_group_tokens,MLLM_dim]")
        n=len(ml_tokens)
        groups=torch.cat([
            torch.full((count,),i,dtype=torch.long,device=ml_tokens.device)
            for i,count in enumerate(self.group_counts)])
        tokens=self.token_projection(ml_tokens.float())+self.segment(groups)[None]
        # Query is learned and shared, while keys/values have image and
        # indicator information from the frozen MLLM joint forward.
        output,_=self.attn(self.query.expand(n,-1,-1),tokens,tokens,
                           need_weights=False)
        return self.fusion(output.squeeze(1))

    def forward(self,visual,context,ml_tokens,mask=None,return_aux=False):
        if context.shape!=(len(visual),4) or len(ml_tokens)!=len(visual):
            raise ValueError("Incompatible RemoteCLIP/context/MLLM batch")
        knowledge=self.read_hidden(ml_tokens)
        h=self.visual(torch.cat((visual.float(),context.float()),dim=-1))
        experts=torch.stack([expert(h) for expert in self.experts],dim=1)
        experts=experts*self.gate(knowledge)[:,None,:]
        logits=self.router(knowledge)
        if mask is not None:
            if mask.shape!=(len(visual),2) or not bool(mask.any(-1).all()):
                raise ValueError("At least one source expert must be active")
            logits=logits.masked_fill(~mask,-1e9)
        weights=F.softmax(logits,dim=-1)
        z=(experts*weights[:,:,None]).sum(dim=1)
        pred=self.out(z).squeeze(-1)
        return (pred,weights,knowledge) if return_aux else pred

    def source_expert_predictions(self,visual,context,ml_tokens):
        # Auxiliary source-expert supervision; does not mask primary router.
        results=[]
        for source in range(2):
            active=torch.zeros((len(visual),2),device=visual.device,dtype=torch.bool)
            active[:,source]=True
            results.append(self(visual,context,ml_tokens,mask=active))
        return torch.stack(results,dim=-1)

    def text_image_alignment(self,knowledge,remoteclip):
        if knowledge.ndim!=2 or remoteclip.ndim!=2 or len(knowledge)!=len(remoteclip):
            raise ValueError("Bad target contrastive shapes")
        # Normalize and match in frozen RemoteCLIP feature space.
        image=F.normalize(remoteclip.float().detach(),dim=-1)
        predicted=F.normalize(self.align(knowledge),dim=-1)
        logits=predicted@image.T/.1
        truth=torch.arange(len(knowledge),device=knowledge.device)
        return (F.cross_entropy(logits,truth)+F.cross_entropy(logits.T,truth))/2
