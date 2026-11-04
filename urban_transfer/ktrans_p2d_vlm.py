"""P2-D-VLM: test actual image+task-conditioned VLM relevance for cross-task expert competence.

P2-B1 used image_task visual_cues and evidence_confidence but OMITTED
image_task evidence_relevance, the ONLY image-specific task-conditioned
six-cue outputs. This experiment adds them to a fixed low-capacity Ridge
cross-task LOPO predictor and tests its genuine conditional information.

No new VLM calls, expert training, target labels or hyperparameter search.
Source LOTO Ridge trains only OTHER pseudo-tasks' SCUI competence.
"""
from collections import defaultdict
import numpy as np
from urban_transfer.ktrans_p2b1 import (
    VARIANTS as OLD_VARIANTS, EVAL_SPLITS, relative_target,
    ridge_predict as baseline_predict, eval_predictions,
    summarize_source_lopo, aggregate)
from urban_transfer.ktrans_p2b0 import cue_permutation

MODES=('task_expert','task_cue_expert',
       'vlm_task_relevance','vlm_task_residual','vlm_task_full',
       'permuted_vlm_task_relevance','permuted_vlm_task_full')

def vlm_components(sample,task_text):
    """The three image_task VLM blocks; no task target measurements.

    Null cues are preserved using separate mask channels.
    text_only conceptual priors are already cached task descriptors.
    """
    visual=np.asarray(sample['visual'],dtype=float)
    masks=np.asarray(sample['mask'],dtype=float)
    if visual.shape!=(3,6) or masks.shape!=(3,6):
        raise ValueError('Expected image_task 18D scores and 18D masks')
    if not np.isfinite(visual).all() or not np.isfinite(masks).all():
        raise ValueError('Invalid VLM knowledge')
    if np.any((visual<0)|(visual>1)) or not np.isin(masks,[0.,1.]).all():
        raise ValueError('Invalid VLM scores or masks')
    if sample['task'] not in task_text:
        raise ValueError('Missing text-only task conceptual relevance')
    prior=np.asarray(task_text[sample['task']],dtype=float)
    if prior.shape!=(6,) or not np.isfinite(prior).all() or np.any((prior<0)|(prior>1)):
        raise ValueError('Invalid text_only relevance')
    scene=visual[0]*masks[0]
    relevance=visual[1]*masks[1]
    confidence=visual[2]*masks[2]
    # Actual VLM task/scene-specific relevance. Do NOT invent
    # observed relevance when the raw VLM field was null.
    conditional=scene*relevance*confidence
    residual=(relevance-prior)*masks[1]
    return scene,relevance,confidence,conditional,residual,prior,masks

def features(sample,expert,source_tasks,task_text,mode):
    if mode not in MODES:raise ValueError('Unsupported VLM knowledge mode')
    tasks=tuple(sorted(source_tasks))
    if sample['task'] not in tasks or expert not in tasks or expert==sample['task']:
        raise ValueError('Incorrect source expert/task identity')
    scene,relevance,confidence,conditional,residual,prior,masks=vlm_components(sample,task_text)
    one=np.zeros(len(tasks),dtype=float);one[tasks.index(expert)]=1.
    # Expert identity globally anchored across pseudo fold columns.
    parts=[np.array([1.]),one]
    if mode=='task_expert':
        parts.append(np.outer(one,prior).reshape(-1))
    elif mode=='task_cue_expert':
        # Exact P2-B1 baseline is dispatched separately in predict()
        raise ValueError('P2-B1 baseline features are not reconstructed here')
    elif mode in ('vlm_task_relevance','permuted_vlm_task_relevance'):
        # Separates visual presence, task relevance and confidence,
        # then includes their multiplicative, task-aware intersection.
        for value in (scene,relevance,confidence,conditional,masks[1]):
            parts.append(np.outer(one,value).reshape(-1))
    elif mode=='vlm_task_residual':
        # VLM's task+image reasoning above/below general text-only prior.
        for value in (prior,scene,confidence,residual,
                      scene*confidence*residual,masks[1]):
            parts.append(np.outer(one,value).reshape(-1))
    elif mode in ('vlm_task_full','permuted_vlm_task_full'):
        for value in (prior,scene,relevance,confidence,
                      conditional,residual,
                      scene*confidence*residual,
                      masks[0],masks[1],masks[2]):
            parts.append(np.outer(one,value).reshape(-1))
    return np.concatenate(parts)

def ridge_predict(train,test,source_tasks,task_text,mode,alpha=10.):
    """Same fixed alpha, target, linear model as original P2-B1."""
    if alpha<=0:raise ValueError('Positive alpha required')
    if mode in OLD_VARIANTS:
        return baseline_predict(train,test,tuple(sorted(source_tasks)),
                                task_text,mode,alpha)
    def matrix(rows,need_y):
        if not rows:raise ValueError('Empty source competence cohort')
        xs=[];ys=[];sizes=[]
        for sample in rows:
            sizes.append(len(sample['experts']))
            for expert in sample['experts']:
                xs.append(features(sample,expert,source_tasks,task_text,mode))
            if need_y:ys.extend(relative_target(sample))
        x=np.asarray(xs,dtype=float)
        if not np.isfinite(x).all():raise ValueError('Nonfinite VLM model features')
        return x,np.asarray(ys,dtype=float),sizes
    x,y,_=matrix(train,True)
    v,_,sizes=matrix(test,False)
    penalty=np.ones(x.shape[1]);penalty[0]=0.
    beta=np.linalg.solve(x.T@x+alpha*np.diag(penalty),x.T@y)
    scores=v@beta
    out=[];start=0
    for size in sizes:
        out.append(scores[start:start+size]);start+=size
    return out

def permute_task_specific_knowledge(samples,rng):
    """Permute ONLY VLM image_task relevance and its observation mask.

    Task, expert identities, visual presence, image confidence, task
    conceptual priors and target labels remain unchanged. The control
    destroys pairing between each tile and the task-conditioned VLM
    relevance, while preserving per-task/split marginal distributions.
    """
    groups=defaultdict(list)
    for i,s in enumerate(samples):
        groups[(s['task'],s['split'])].append(i)
    result=list(samples)
    for key in sorted(groups):
        idx=groups[key]
        order=rng.permutation(len(idx))
        for dest,source_offset in zip(idx,order):
            donor=samples[idx[int(source_offset)]]
            visual=np.asarray(samples[dest]['visual'],dtype=float).copy()
            masks=np.asarray(samples[dest]['mask'],dtype=float).copy()
            visual[1]=donor['visual'][1]
            masks[1]=donor['mask'][1]
            result[dest]={**samples[dest],'visual':visual,'mask':masks}
    return result

def run_cross_task(samples,source_tasks,task_text,
                   alpha=10.,temperature=.5,seed=42):
    """Five source LOPO folds; test withheld pseudo-task SCUI and UCUI."""
    tasks=tuple(sorted(source_tasks))
    if set(task_text)!=set(tasks):raise ValueError('Missing source task text_only cache')
    rows=[]
    for hold_idx,holdout in enumerate(tasks):
        train=[s for s in samples if s['task']!=holdout and s['split']=='meta_SCUI']
        for split_idx,split in enumerate(EVAL_SPLITS):
            test=[s for s in samples if s['task']==holdout and s['split']==split]
            if not train or not test:raise ValueError('Missing source task fold/split')
            for mode in MODES:
                if mode.startswith('permuted_vlm_'):
                    rng=np.random.default_rng(seed+hold_idx*100+split_idx)
                    tr=permute_task_specific_knowledge(train,rng)
                    te=permute_task_specific_knowledge(test,rng)
                else:tr,te=train,test
                scores=ridge_predict(tr,te,tasks,task_text,mode,alpha)
                for metric in eval_predictions(test,scores,temperature):
                    metric['model']=mode
                    metric['heldout_source_task']=holdout
                    rows.append(metric)
    return rows

def run_same_task(samples,source_tasks,task_text,
                  alpha=10.,temperature=.5,seed=42):
    """Supervised SCUI→UCUI diagnostic only; NOT cross-task zero-shot."""
    tasks=tuple(sorted(source_tasks))
    rows=[]
    for task_idx,task in enumerate(tasks):
        train=[s for s in samples if s['task']==task and s['split']=='meta_SCUI']
        test=[s for s in samples if s['task']==task and s['split']=='meta_UCUI']
        if not train or not test:raise ValueError('Incomplete same-task source city split')
        if {s['city'] for s in train}&{s['city'] for s in test}:
            raise ValueError('SCUI and UCUI cities overlap')
        for mode in MODES:
            if mode.startswith('permuted_vlm_'):
                rng=np.random.default_rng(seed+9000+task_idx)
                tr=permute_task_specific_knowledge(train,rng)
                te=permute_task_specific_knowledge(test,rng)
            else:tr,te=train,test
            scores=ridge_predict(tr,te,tasks,task_text,mode,alpha)
            for metric in eval_predictions(test,scores,temperature):
                metric['model']=mode
                metric['heldout_source_task']=task
                metric['route']='same_task_scui_to_ucui'
                rows.append(metric)
    return rows

def summarize_same_task(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['model'],row['pseudo_task']].append(row)
    task_rows=[{'model':mode,'pseudo_task':task,**aggregate(group)}
               for (mode,task),group in sorted(groups.items())]
    macro=defaultdict(list)
    for r in task_rows:macro[r['model']].append(r)
    summaries=[]
    for mode,vals in sorted(macro.items()):
        summaries.append({'model':mode,'n_source_pseudo_tasks':len(vals),
            'n_tiles':sum(r['n_tiles'] for r in vals),
            'macro_top1_accuracy':float(np.mean([r['top1_fractional_accuracy'] for r in vals])),
            'macro_pairwise_accuracy':float(np.mean([r['pairwise_accuracy'] for r in vals])),
            'macro_mse_ratio_to_uniform':float(np.nanmean([r['soft_mix_mse_ratio_to_uniform'] for r in vals]))})
    return task_rows,summaries

def vlm_signal_audit(samples,task_text):
    """Task/split VLM relevance variability and genuine image-specific residuals.

    If conditional relevance is nearly constant or always equal to the
    text-only prior, a cross-task router cannot learn useful per-region
    distinctions from this six-cue channel.
    """
    groups=defaultdict(list)
    for s in samples:groups[s['task'],s['split']].append(s)
    rows=[]
    for (task,split),group in sorted(groups.items()):
        for cue in range(6):
            observed=[];deltas=[]
            for s in group:
                _,rel,_,_,delta,prior,masks=vlm_components(s,task_text)
                if masks[1,cue]:
                    observed.append(float(rel[cue]))
                    deltas.append(float(delta[cue]))
            arr=np.asarray(observed,dtype=float)
            d=np.asarray(deltas,dtype=float)
            rows.append({'pseudo_task':task,'inner_partition':split,
                'cue_index':cue,'n_tiles':len(group),
                'n_relevance_observed':len(arr),
                'n_unique_relevance_scores':len(set(arr.tolist())),
                'relevance_variance':float(np.var(arr)) if len(arr) else float('nan'),
                'mean_abs_image_task_minus_text_only':float(np.mean(np.abs(d))) if len(d) else float('nan'),
                'fraction_equal_to_text_only_prior':float(np.mean(np.abs(d)<1e-12)) if len(d) else float('nan')})
    return rows
