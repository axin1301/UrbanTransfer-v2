"""K7 M5: concept-grounded rank-wise LoRA composition.

Each of 4 visual concepts owns 2 ranks (rank=8). Concept identities are
explicit and grounded by frozen RemoteCLIP image/text similarities,
rather than guessed from arbitrary LoRA dimensions.
"""
import torch
import torch.nn.functional as F
from torch import nn
from urban_transfer.k7_lora_composer import RelationLoRAComposer

CONCEPT_COUNT=4

class ConceptLoRAComposer(RelationLoRAComposer):
    def __init__(self,visual_dim,semantic_dim,hidden=128,rank=8):
        if rank%CONCEPT_COUNT:
            raise ValueError("Concept-grounded LoRA rank must be divisible by 4")
        super().__init__(visual_dim,semantic_dim,hidden=hidden,rank=rank)
        # This generator learns a separate gate for every grounded rank.
        # Remove M3's unconstrained source attention in forward below.
        self.rank_gate=nn.Linear(hidden,rank)
        nn.init.zeros_(self.rank_gate.weight)
        nn.init.zeros_(self.rank_gate.bias)

    def forward(self,visual,context,target_semantic,source_semantics,
                edge_relations,self_edges,edge_mask=None,
                return_aux=False,return_concepts=False):
        n=len(visual)
        if context.shape!=(n,4) or target_semantic.shape[0]!=n:
            raise ValueError("Bad visual input")
        if source_semantics.ndim==2:
            source_semantics=source_semantics.unsqueeze(0).expand(n,-1,-1)
        shape=(n,2,target_semantic.shape[-1])
        if source_semantics.shape!=shape or edge_relations.shape!=shape or self_edges.shape!=(n,2):
            raise ValueError("M5 requires two distinct source relation vectors")
        if edge_mask is None:
            edge_mask=torch.ones(n,2,dtype=torch.bool,device=visual.device)
        if edge_mask.shape!=(n,2) or not bool(edge_mask.any(-1).all()):
            raise ValueError("At least one active source needed")
        h=self.trunk(torch.cat((visual.float(),context.float()),dim=-1))
        t=F.normalize(target_semantic.float(),dim=-1)
        s=F.normalize(source_semantics.float(),dim=-1)
        e=F.normalize(edge_relations.float(),dim=-1)
        edge=self.edge(torch.cat((
            s,t.unsqueeze(1).expand(-1,2,-1),e,
            self_edges.float().unsqueeze(-1)),dim=-1))
        target_h=self.task(t)
        # Softmax gates: every source's per-rank weights sum to rank.
        gates=F.softmax(self.rank_gate(edge),dim=-1)*self.rank
        # The same rank positions are weakly aligned to CLIP concept anchors.
        activations=torch.einsum("nh,srh->nsr",h,self.source_a)
        updates=torch.einsum(
            "nsr,shr->nsh",activations*gates,self.source_b)/self.rank
        # NO trainable source-level router: prevents the M3/M4 collapse.
        mix=edge_mask.float()/edge_mask.float().sum(-1,keepdim=True)
        adapted=h+self.gain*(updates*mix.unsqueeze(-1)).sum(dim=1)
        pred=(self.readout(adapted)+self.bias(target_h)).squeeze(-1)
        if return_concepts:
            return pred,mix,gates,activations
        if return_aux:
            return pred,mix,gates
        return pred


def rank_targets(relation_embeddings,concept_embeddings,temperature=0.15,
                 rank=8):
    """Qwen semantic teacher: relation-to-concept weights, not source weights."""
    if (relation_embeddings.ndim!=2 or relation_embeddings.shape[0]!=2
        or concept_embeddings.ndim!=2 or concept_embeddings.shape[0]!=4
        or relation_embeddings.shape[1]!=concept_embeddings.shape[1]
        or rank%4 or temperature<=0):
        raise ValueError("Invalid teacher text shapes/temperature/rank")
    similarity=F.normalize(relation_embeddings.float(),dim=-1) @ F.normalize(
        concept_embeddings.float(),dim=-1).T
    distribution=F.softmax(similarity/temperature,dim=-1)
    # Fixed repeat mapping: slots 0-1 = residential, 2-3 = settlement,
    # 4-5 = roads, 6-7 = commercial built-up environment.
    distribution=distribution.repeat_interleave(rank//4,dim=-1)/(rank//4)
    return distribution.detach()


def concept_supervision(activations,clip_target,rank=8):
    """Weak visual grounding: RemoteCLIP image/concept cosine similarities."""
    if (activations.ndim!=3 or activations.shape[1:]!=(2,rank) or
        clip_target.shape!=(activations.shape[0],4) or rank%4):
        raise ValueError("Concept alignment needs [B,2,R] and [B,4]")
    labels=clip_target.repeat_interleave(rank//4,dim=-1)
    return F.mse_loss(torch.sigmoid(activations),labels[:,None,:].expand(-1,2,-1))


def rank_distillation(student_gates,teacher_distribution,rank=8):
    """Teacher [2,R] supervises both source-conditioned rank allocations."""
    if (student_gates.ndim!=3 or student_gates.shape[1:]!=(2,rank)
        or teacher_distribution.shape!=(2,rank)):
        raise ValueError("Wrong student/teacher rank shapes")
    teacher=teacher_distribution.to(student_gates.device,student_gates.dtype)
    return F.kl_div((student_gates/rank).clamp_min(1e-8).log(),
        teacher[None].expand_as(student_gates),reduction="batchmean")/2
