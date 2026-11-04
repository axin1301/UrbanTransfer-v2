"""Source-only cue-to-expert alignment prototype (NOT a validated model).

Pseudo-task transfer scores are obtained from K5-A INNER heldout folds,
never from the outer target. Each source expert receives an interpretable
six-cue proxy profile from positive transfer association with other
source tasks' text-only cue relevance. This is an identification
hypothesis, not measured causal expertise.
"""
import csv
import numpy as np
from urban_transfer.ktrans_k7d import CUES

def cue_relevance(parsed):
    scores=parsed.get('evidence_relevance')
    if not isinstance(scores,dict):raise ValueError('Missing text-only conceptual cue relevance')
    vals=[]
    for cue in CUES:
        value=scores.get(cue)
        if value is None:
            vals.append(0.)
        elif type(value) is int and 0<=value<=3:
            vals.append(value/3.)
        else:
            raise ValueError('Invalid conceptual cue relevance')
    return np.asarray(vals,dtype=np.float64)

def image_task_demand(parsed):
    """Use observed morphology and task-conditioned proxy relevance.

    Unobserved/null confidence is treated as unknown: zero contribution,
    NOT a fabricated score. Retain no target indicator values.
    """
    blocks=[]
    for field in ('visual_cues','evidence_relevance','evidence_confidence'):
        data=parsed.get(field)
        if not isinstance(data,dict):raise ValueError('Missing '+field)
        arr=[]
        for cue in CUES:
            val=data.get(cue)
            if val is None:arr.append(0.)
            elif type(val) is int and 0<=val<=3:arr.append(val/3.)
            else:raise ValueError('Invalid '+field+' score')
        blocks.append(np.asarray(arr,dtype=np.float64))
    # Confidence is a reliability gate; the result is a task/scene hypothesis.
    return blocks[0]*blocks[1]*blocks[2]

def build_expert_cue_proxy(source_tasks,score_rows,text_relevance,minimum_evidence=2):
    source_tasks=tuple(source_tasks)
    if len(source_tasks)!=len(set(source_tasks)):raise ValueError('Duplicated source tasks')
    if set(text_relevance)!=set(source_tasks):
        raise ValueError('Task-only knowledge must cover exactly the source tasks')
    rel={task:cue_relevance(text_relevance[task]) for task in source_tasks}
    found=set();samples={s:[] for s in source_tasks}
    for row in score_rows:
        pseudo=row['pseudo_target'];source=row['source_expert']
        if pseudo not in source_tasks or source not in source_tasks or pseudo==source:
            raise ValueError('Score row is not an inner source-only pseudo-holdout pair')
        key=(pseudo,source)
        if key in found:raise ValueError('Duplicate pseudo/subject source score')
        found.add(key)
        val=float(row['spatial_pearson'])
        if not np.isfinite(val) or not -1<=val<=1:
            raise ValueError('Invalid pseudo-transfer correlation')
        samples[source].append((pseudo,max(0.,val)))
    result={}
    for source in source_tasks:
        pairs=samples[source]
        if len(pairs)<minimum_evidence:raise ValueError('Insufficient nested fold evidence for '+source)
        confidence=np.asarray([v for _,v in pairs],dtype=np.float64)
        if confidence.sum()<=1e-10:raise ValueError('No positive transfer evidence for '+source)
        # Weighted averages of conceptual cue relevance, not empirical
        # source-expert sensitivity or a learned target-specific feature.
        profile=sum(v*rel[t] for t,v in pairs)/confidence.sum()
        result[source]={'cue_proxy':profile.tolist(),'positive_transfer_weight':float(confidence.sum()),
                        'n_inner_pairs':len(pairs),'supporting_pseudo_tasks':[t for t,_ in pairs]}
    return result

def alignment_route(source_tasks,expert_proxy,target_image_task,anchor,
                    strength=0.25,temperature=0.25):
    """Fixed unsupervised convex mixture; no target-label optimization."""
    if not (0<=strength<=1) or temperature<=0:
        raise ValueError('Invalid predeclared alignment strength/temperature')
    tasks=tuple(source_tasks)
    if set(expert_proxy)!=set(tasks):
        raise ValueError('Expert proxy ordering/membership mismatch')
    a=np.asarray(anchor,dtype=np.float64)
    if a.shape!=(len(tasks),) or not np.isfinite(a).all() or (a<0).any() or a.sum()<=0:
        raise ValueError('Invalid frozen K5-A anchor')
    a=a/a.sum()
    q=image_task_demand(target_image_task)
    # Uninformative visual relevance leaves the anchor unchanged.
    if np.linalg.norm(q)<=1e-12:return a.copy(),{'cue_evidence':0.,'fallback':'no_confident_visual_evidence'}
    q=q/(np.linalg.norm(q)+1e-12)
    p=np.stack([np.asarray(expert_proxy[t]['cue_proxy'],dtype=np.float64) for t in tasks])
    if p.shape!=(len(tasks),len(CUES)) or not np.isfinite(p).all() or (p<0).any():
        raise ValueError('Invalid cue proxy')
    norm=np.linalg.norm(p,axis=1,keepdims=True)
    if (norm<=1e-12).any():raise ValueError('Zero source cue proxy')
    similarity=(p/norm)@q
    logits=similarity/temperature
    logits=logits-logits.max()
    aligned=np.exp(logits);aligned=aligned/aligned.sum()
    route=(1-strength)*a+strength*aligned
    return route,{'cue_evidence':float(np.linalg.norm(image_task_demand(target_image_task))),
                  'fallback':None,'alignment_similarity':similarity.tolist(),
                  'aligned_route':aligned.tolist()}

def read_transfer_scores(path):
    with open(path,newline='') as f:
        return list(csv.DictReader(f))
