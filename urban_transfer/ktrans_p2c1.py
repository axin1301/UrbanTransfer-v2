"""P2-C1: source pseudo-task conditional expert competence structure audit.

The central tensor is task × named-expert × six VLM visual cues.
Diagonal (pseudo task == expert) is NOT observed and is never zero-imputed.
Effects are high-minus-low relative squared loss from P2-A, using
ORIGINAL SCUI-only threshold per task/cue. Descriptive DEV only.
"""
from itertools import combinations
import numpy as np
from urban_transfer.ktrans_k7d import CUES
from urban_transfer.ktrans_p2a import task_cue_thresholds,condition_rows

def standardize_effects(samples,source_tasks,min_group=8):
    """Create observed effect tensor and mask with per-task scale normalization.

    All entries are constructed separately by split. SCUI source-task
    within-tile loss scale only; UCUI never sets scale or thresholds.
    This controls larger-label pseudo tasks dominating similarities.
    """
    names=tuple(sorted(source_tasks))
    thresholds=task_cue_thresholds(samples)
    grouped={t:[s for s in samples if s['task']==t and s['split']=='meta_SCUI'] for t in names}
    scales={}
    for task,rows in grouped.items():
        if not rows:raise ValueError('Missing SCUI pseudo task '+task)
        # Robust SCUI-only reference spread of expert losses in same tile;
        # not task label variance and never UCUI calibrated.
        span=np.array([np.max(s['loss'])-np.min(s['loss']) for s in rows])
        nonzero=span[span>1e-12]
        scales[task]=float(np.median(nonzero)) if len(nonzero) else 1.
        scales[task]=max(scales[task],1e-8)
    condition=condition_rows(samples,thresholds,min_group)
    tensor={};mask={};detail=[]
    for split in ('meta_SCUI','meta_UCUI'):
        cube=np.full((len(names),len(names),len(CUES)),np.nan,dtype=float)
        observed=np.zeros_like(cube,dtype=bool)
        for row in condition:
            if row['inner_partition']!=split:continue
            t=row['pseudo_task'];e=row['expert'];c=row['cue']
            if t not in names or e not in names or t==e:
                raise ValueError('Contaminated source conditional effect')
            value=float(row['high_minus_low_relative_loss'])
            if not row['enough_samples'] or not np.isfinite(value):continue
            i=names.index(t);j=names.index(e);k=CUES.index(c)
            scaled=value/scales[t]
            cube[i,j,k]=scaled;observed[i,j,k]=True
            detail.append({'pseudo_task':t,'inner_partition':split,
                'expert':e,'cue':c,'n_low':row['n_low'],
                'n_high':row['n_high'],'n_tied':row['n_tied'],
                'raw_effect_high_minus_low':value,
                'scui_reference_scale':scales[t],
                'normalized_effect':scaled})
        tensor[split]=cube;mask[split]=observed
    return names,tensor,mask,detail,thresholds,scales

def masked_cosine(a,b,valid):
    if valid.sum()<2:return float('nan')
    x=a[valid];y=b[valid]
    norm=float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(x@y/norm) if norm>1e-12 else float('nan')

def task_pair_structure(names,tensor,masks,task_vectors=None):
    """Compare task pairs only on the three common named experts.

    A five-task bank has four experts per fold. Each pair shares
    exactly three; no synthetic own-expert capabilities are filled.
    """
    comparisons=[]
    for split in ('meta_SCUI','meta_UCUI'):
        cube=tensor[split];present=masks[split]
        for i,j in combinations(range(len(names)),2):
            selected=np.array([k for k in range(len(names)) if k not in (i,j)])
            a=cube[i,selected,:].reshape(-1)
            b=cube[j,selected,:].reshape(-1)
            valid=(present[i,selected,:]&present[j,selected,:]).reshape(-1)
            cosine=masked_cosine(a,b,valid)
            signs=float(np.mean(np.sign(a[valid])==np.sign(b[valid]))) if valid.any() else float('nan')
            text_sim=float('nan')
            if task_vectors is not None:
                u=np.asarray(task_vectors[names[i]],dtype=float)
                v=np.asarray(task_vectors[names[j]],dtype=float)
                if u.shape!=v.shape or u.ndim!=1:raise ValueError('Wrong source text descriptor shapes')
                text_sim=masked_cosine(u,v,np.isfinite(u)&np.isfinite(v))
            comparisons.append({'inner_partition':split,
                'task_a':names[i],'task_b':names[j],
                'n_common_named_experts':len(selected),
                'n_effect_entries_shared':int(valid.sum()),
                'competence_effect_cosine':cosine,
                'competence_effect_sign_agreement':signs,
                'text_only_cue_cosine':text_sim})
    return comparisons

def rank_structure(names,tensor,masks,split='meta_SCUI'):
    """SVD of observed complete rectangular submatrices only.

    Missing pseudo-own expert blocks preclude a full 5×30 SVD.
    For each pair of pseudo tasks, use common named experts (3×6)
    and only cues observed for both rows. Report descriptive
    low-rank energy, NOT a heldout-task generalization estimate.
    """
    a=tensor[split];m=masks[split]
    results=[]
    for i,j in combinations(range(len(names)),2):
        experts=[k for k in range(len(names)) if k not in (i,j)]
        entries=np.array([a[i,experts,:].reshape(-1),a[j,experts,:].reshape(-1)])
        good=np.array([m[i,experts,:].reshape(-1),m[j,experts,:].reshape(-1)]).all(0)
        x=entries[:,good]
        if x.shape[1]<2:singular=np.array([])
        else:singular=np.linalg.svd(x,full_matrices=False,compute_uv=False)
        energy=(singular**2)/np.sum(singular**2) if len(singular) and np.sum(singular**2)>1e-12 else np.array([])
        results.append({'inner_partition':split,'task_a':names[i],
            'task_b':names[j],'n_common_entries':int(good.sum()),
            'rank1_energy_fraction':float(energy[0]) if len(energy) else float('nan'),
            'n_available_singular_values':len(singular),
            'warning':'Two-task complete block only; rank1 energy is descriptive, NOT full-tensor low-rank proof'})
    return results

def task_semantic_association(rows):
    """At most 10 task pairs: correlation is descriptive, not inference."""
    x=np.asarray([r['text_only_cue_cosine'] for r in rows],dtype=float)
    y=np.asarray([r['competence_effect_cosine'] for r in rows],dtype=float)
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.std(x[ok])<1e-12 or np.std(y[ok])<1e-12:
        return {'n_pairs':int(ok.sum()),'pearson':float('nan')}
    return {'n_pairs':int(ok.sum()),'pearson':float(np.corrcoef(x[ok],y[ok])[0,1])}
