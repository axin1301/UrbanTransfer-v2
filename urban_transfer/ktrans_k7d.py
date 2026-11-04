"""K7-D lightweight, anchored expert-transfer router with frozen VLM knowledge.

The candidate route is learned only from outer source-task rows, masking
the current task's own expert. Frozen K4/K5-A provide the expert bank and
transfer prior; semantic source moments set the unseen target scale.
This is a DEVELOPMENT prototype, not independently cross-fitted evidence.
"""
import torch
import torch.nn.functional as F
from torch import nn

CUES=('built_up','roads','residential','vegetation','large_facilities','industrial_commercial')
VARIANTS=('k5a','text_only','image_only','image_task')
FEATURE_DIM=18  # six ordinal scores for each of three groups

def encode_record(parsed,mode):
    """Return normalized six-cue scores and separate observation masks.

    Nulls are encoded as 0 with masks=0, never conflated with a true
    ordinal-0 observation. A model cannot infer cue presence from text_only.
    """
    if parsed is None:raise ValueError('Missing/invalid VLM JSON')
    fields=('visual_cues','evidence_relevance','evidence_confidence')
    values=[];masks=[]
    for field in fields:
        table=parsed.get(field)
        if mode=='image_only' and field=='evidence_relevance' and table is None:
            table={cue:None for cue in CUES}
        if not isinstance(table,dict):raise ValueError('Missing VLM evidence field: '+field)
        for cue in CUES:
            val=table.get(cue)
            if val is not None and (type(val) is not int or val not in (0,1,2,3)):
                raise ValueError('Non-ordinal VLM score')
            values.append(0. if val is None else val/3.)
            masks.append(0. if val is None else 1.)
    if mode=='text_only':
        if any(masks[:6]) or any(masks[12:]):
            raise ValueError('Text-only record pretends to observe visual evidence')
    if mode=='image_only' and any(masks[6:12]):
        raise ValueError('Image-only record contains task relevance')
    return torch.tensor(values+masks,dtype=torch.float32)

def knowledge_vector(image_task,image_only,text_only,variant):
    """All non-baseline ablations have 36 dimensions.

    Text-only: conceptual task prior only.
    Image-only: scene morphology only.
    Image-task: task-conditioned evidence plus independent controls.
    """
    z=torch.zeros(36,dtype=torch.float32)
    if variant=='k5a':return z
    if variant=='text_only':
        return encode_record(text_only,'text_only')
    if variant=='image_only':
        return encode_record(image_only,'image_only')
    if variant=='image_task':
        v=encode_record(image_task,'image_task')
        # Separate task/visual effects through a residual to the
        # conceptual text prior; retain VLM evidence/missingness.
        prior=encode_record(text_only,'text_only')
        v[6:12]-=prior[6:12]
        return v
    raise ValueError('Unknown ablation '+str(variant))

class AnchoredKnowledgeRouter(nn.Module):
    """Task-aware low-rank knowledge adjustment bounded by a global K5-A route."""
    def __init__(self,n_sources,feature_dim=36,hidden=24,init_gate=-2.):
        super().__init__()
        self.n=n_sources
        self.net=nn.Sequential(nn.Linear(feature_dim,hidden),nn.GELU(),
                               nn.Linear(hidden,n_sources))
        self.strength=nn.Sequential(nn.Linear(feature_dim,hidden),nn.GELU(),
                                    nn.Linear(hidden,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.zeros_(self.strength[-1].weight)
        nn.init.constant_(self.strength[-1].bias,init_gate)

    def forward(self,knowledge,prior,source_mask=None):
        if knowledge.ndim!=2 or prior.ndim!=2 or prior.shape[1]!=self.n or len(prior)!=len(knowledge):
            raise ValueError('Incompatible router inputs')
        if source_mask is None:source_mask=torch.zeros_like(prior,dtype=torch.bool)
        if source_mask.shape!=prior.shape:
            raise ValueError('Mask must match source expert count')
        p=prior.masked_fill(source_mask,0.)
        if (p.sum(-1)<=0).any():raise ValueError('All source experts masked')
        p=p/p.sum(-1,keepdim=True)
        raw=p.clamp_min(1e-12).log()+self.net(knowledge)
        logits=raw.masked_fill(source_mask,-1e4)
        candidate=F.softmax(logits,dim=-1)
        gate=torch.sigmoid(self.strength(knowledge)).flatten()
        weights=p*(1-gate[:,None])+candidate*gate[:,None]
        return weights,gate

def z_predict(expert_outputs,weights,mean,std):
    if expert_outputs.shape!=weights.shape:raise ValueError('Expert outputs must align with weights')
    return mean+std*(expert_outputs*weights).sum(-1)
