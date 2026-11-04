"""P2-B1 low-capacity source-only cross-pseudo-task competence predictor.

Each SOURCE pseudo-task was held out from its INNER expert bank. Ridge
fits only SCUI labels of OTHER pseudo-tasks. This module never loads
outer target labels or adjusts a model against source UCUI outcomes.
All experts are named globally, not by changing fold-specific columns.
"""
from collections import defaultdict
import numpy as np
from urban_transfer.ktrans_k7d import CUES
from urban_transfer.ktrans_p2a import samples_from_rows
from urban_transfer.ktrans_p2b0 import cue_permutation

N_CUES=len(CUES)
VARIANTS=('task_expert','cue_expert','task_cue_expert','permuted_task_cue_expert')
EVAL_SPLITS=('meta_SCUI','meta_UCUI')

def aligned_samples(rows,outer_target,source_tasks):
    samples=samples_from_rows(rows,outer_target)
    lookup=defaultdict(dict)
    for row in rows:
        key=(row['pseudo_task'],row['inner_partition'],row['tile_id'])
        lookup[key][row['expert']]=row
    source=set(source_tasks)
    for sample in samples:
        if sample['task'] not in source or set(sample['experts'])!=source-{sample['task']}:
            raise ValueError('Unexpected pseudo-task/expert identities')
        original=lookup[(sample['task'],sample['split'],sample['tile'])]
        # Work on log1p label scale; frozen inner source-only semantic moments.
        sample['pred']=np.asarray([float(original[e]['prediction_log1p']) for e in sample['experts']])
        sample['target']=float(original[sample['experts'][0]]['label_log1p'])
        if not np.isfinite(sample['pred']).all() or not np.isfinite(sample['target']):
            raise ValueError('Nonfinite prediction or source pseudo-label')
        if not np.allclose((sample['pred']-sample['target'])**2,sample['loss'],
                           rtol=1e-5,atol=1e-7):
            raise ValueError('Source pseudo-task predictions/labels/loss mismatch')
    return samples

def relative_target(sample):
    """Bounded within-tile expert regret: no cross-task loss scale leakage.

    Train labels only for SCUI of other SOURCE pseudo-tasks. At inference
    target losses are never used to compute model features or weights.
    """
    loss=sample['loss']
    span=float(loss.max()-loss.min())
    if span<1e-12:return np.zeros_like(loss)
    return (loss-loss.mean())/span

def sample_features(sample,expert,source_tasks,task_text,variant):
    if expert not in source_tasks or expert==sample['task']:
        raise ValueError('Unknown or pseudo-own source expert')
    if sample['task'] not in task_text:raise ValueError('Missing text-only source task descriptor')
    relevant=np.asarray(task_text[sample['task']],dtype=float)
    if relevant.shape!=(N_CUES,) or not np.isfinite(relevant).all():
        raise ValueError('Invalid text-only conceptual task descriptor')
    if np.any((relevant<0)|(relevant>1)):raise ValueError('Invalid cue relevance')
    scene=sample['visual'][0]
    mask=sample['mask'][0]
    confidence=sample['visual'][2]*sample['mask'][2]
    # All image signals carry presence/confidence; absent/null is not
    # silently treated as observed zero.
    scene=np.asarray(scene*mask,dtype=float)
    evidence=np.asarray(scene*confidence,dtype=float)
    if variant not in VARIANTS:raise ValueError('Unknown P2-B1 variant')
    # Fixed dimensions for every expert. Global IDs remain meaningful
    # when a different pseudo task is held out in each inner fold.
    one=np.zeros(len(source_tasks))
    one[source_tasks.index(expert)]=1.
    pieces=[np.array([1.]),one]
    if variant in ('task_expert','task_cue_expert','permuted_task_cue_expert'):
        pieces.append(np.outer(one,relevant).reshape(-1))
    if variant!='task_expert':
        pieces.extend([np.outer(one,scene).reshape(-1),
                       np.outer(one,mask).reshape(-1)])
    if variant in ('task_cue_expert','permuted_task_cue_expert'):
        pieces.extend([np.outer(one,evidence*relevant).reshape(-1)])
    return np.concatenate(pieces)

def matrix(samples,source_tasks,task_text,variant,targets):
    X=[];Y=[];sizes=[]
    for sample in samples:
        sizes.append(len(sample['experts']))
        for expert in sample['experts']:
            X.append(sample_features(sample,expert,source_tasks,task_text,variant))
        if targets:Y.extend(relative_target(sample))
    X=np.asarray(X,dtype=np.float64)
    if not np.isfinite(X).all():raise ValueError('Nonfinite competence features')
    return X,(np.asarray(Y,dtype=np.float64) if targets else None),sizes

def ridge_predict(train,test,source_tasks,task_text,variant,alpha):
    if alpha<=0:raise ValueError('Ridge penalty must be positive')
    X,y,_=matrix(train,source_tasks,task_text,variant,True)
    V,_,sizes=matrix(test,source_tasks,task_text,variant,False)
    # All coefficients are ridge-shrunk except the intercept. Work in
    # dual space only if fewer samples than features.
    penalty=np.ones(X.shape[1]);penalty[0]=0.
    coef=np.linalg.solve(X.T@X+alpha*np.diag(penalty),X.T@y)
    scores=V@coef
    out=[];offset=0
    for count in sizes:
        out.append(scores[offset:offset+count])
        offset+=count
    return out

def soft_route(scores,temperature):
    if temperature<=0:raise ValueError('Temperature must be positive')
    logits=-np.asarray(scores,dtype=float)/temperature
    logits-=logits.max()
    vals=np.exp(logits)
    return vals/vals.sum()

def eval_predictions(samples,predicted,temperature):
    """Tile-unit metrics, aggregate error on SOURCE pseudo labels.

    No oracle exposure in weight inference: oracle regret reported
    separately for descriptive ground-truth comparison.
    """
    if len(samples)!=len(predicted) or not samples:raise ValueError('Misaligned predictions')
    metrics=[]
    for sample,scores in zip(samples,predicted):
        if len(scores)!=len(sample['experts']):raise ValueError('Expert score shape mismatch')
        gt=sample['loss'];scores=np.asarray(scores,dtype=float)
        if not np.isfinite(scores).all():raise ValueError('Nonfinite competence scores')
        weights=soft_route(scores,temperature)
        predicted_best=int(np.argmin(scores))
        actual_best=np.flatnonzero(np.isclose(gt,gt.min(),rtol=1e-10,atol=1e-12))
        pair_correct=[];pair_total=0
        for i in range(len(gt)):
            for j in range(i+1,len(gt)):
                if abs(gt[i]-gt[j])<1e-12:continue
                if abs(scores[i]-scores[j])<1e-12:pair_correct.append(.5)
                else:pair_correct.append(float((scores[i]-scores[j])*(gt[i]-gt[j])>0))
                pair_total+=1
        blend=float(weights@sample['pred'])
        uniform=float(np.mean(sample['pred']))
        # The oracle single expert is an unrealizable label-dependent upper bound.
        metrics.append({'pseudo_task':sample['task'],'inner_partition':sample['split'],
            'city':sample['city'],'tile_id':sample['tile'],
            'n_experts':len(gt),'top1_fractional_correct':float(predicted_best in actual_best)/len(actual_best),
            'pairwise_correct_sum':float(sum(pair_correct)),
            'pairwise_n':pair_total,
            'selected_regret':float(gt[predicted_best]-gt.min()),
            'oracle_min_squared_error':float(gt.min()),
            'selected_expert_squared_error':float(gt[predicted_best]),
            'soft_mix_squared_error':(blend-sample['target'])**2,
            'uniform_mix_squared_error':(uniform-sample['target'])**2,
            'soft_mix_prediction':blend,'uniform_mix_prediction':uniform,
            'target_log1p':sample['target']})
    return metrics

def aggregate(items):
    """Equal weight to each tile within a SOURCE pseudo-task / split."""
    if not items:raise ValueError('No unit metrics')
    soft_mse=float(np.mean([r['soft_mix_squared_error'] for r in items]))
    uniform_mse=float(np.mean([r['uniform_mix_squared_error'] for r in items]))
    return {'n_tiles':len(items),'n_cities':len({r['city'] for r in items}),
        'top1_fractional_accuracy':float(np.mean([r['top1_fractional_correct'] for r in items])),
        'pairwise_accuracy':float(sum(r['pairwise_correct_sum'] for r in items)/
            max(1,sum(r['pairwise_n'] for r in items))),
        'mean_selected_regret':float(np.mean([r['selected_regret'] for r in items])),
        'selected_expert_mse':float(np.mean([r['selected_expert_squared_error'] for r in items])),
        'oracle_single_expert_mse':float(np.mean([r['oracle_min_squared_error'] for r in items])),
        'soft_mix_mse':soft_mse,'uniform_mix_mse':uniform_mse,
        'soft_mix_mse_ratio_to_uniform':float(soft_mse/uniform_mse) if uniform_mse>1e-12 else float('nan')}

def source_lopo(samples,source_tasks,task_text,alpha=10.,temperature=.5,seed=42):
    """Five honest SOURCE pseudo-task-heldout folds, SCUI fit only.

    SCUI on withheld pseudo task is a source cross-task evaluation.
    UCUI on withheld pseudo task is exploratory cross-task AND unseen
    inner-city evaluation; neither is used for tuning any setting.
    """
    tasks=tuple(sorted(source_tasks))
    if set(task_text)!=set(tasks):raise ValueError('Text-only source task coverage mismatch')
    outputs=[]
    for holdout in tasks:
        train=[s for s in samples if s['task']!=holdout and s['split']=='meta_SCUI']
        if not train:raise ValueError('Empty source-only LOTO training fold')
        for split in EVAL_SPLITS:
            test=[s for s in samples if s['task']==holdout and s['split']==split]
            if not test:raise ValueError('Empty source-only LOTO evaluation fold')
            for variant in VARIANTS:
                if variant=='permuted_task_cue_expert':
                    rng=np.random.default_rng(seed+tasks.index(holdout)*100+EVAL_SPLITS.index(split))
                    tr,_=cue_permutation(train,rng,within_city=False)
                    te,_=cue_permutation(test,rng,within_city=False)
                else:tr,te=train,test
                scores=ridge_predict(tr,te,tasks,task_text,variant,alpha)
                for metric in eval_predictions(test,scores,temperature):
                    metric['model']=variant
                    metric['heldout_source_task']=holdout
                    outputs.append(metric)
    return outputs

def summarize_source_lopo(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['model'],row['inner_partition'],row['pseudo_task'])].append(row)
    task_rows=[]
    for (model,split,task),vals in sorted(groups.items()):
        task_rows.append({'model':model,'inner_partition':split,
                          'pseudo_task':task,**aggregate(vals)})
    macro=defaultdict(list)
    for row in task_rows:macro[(row['model'],row['inner_partition'])].append(row)
    summaries=[]
    for (model,split),task_stats in sorted(macro.items()):
        summaries.append({'model':model,'inner_partition':split,
            'n_source_pseudo_tasks':len(task_stats),
            'n_tiles':sum(t['n_tiles'] for t in task_stats),
            'macro_top1_accuracy':float(np.mean([t['top1_fractional_accuracy'] for t in task_stats])),
            'macro_pairwise_accuracy':float(np.mean([t['pairwise_accuracy'] for t in task_stats])),
            'macro_selected_regret':float(np.mean([t['mean_selected_regret'] for t in task_stats])),
            'macro_soft_mix_mse':float(np.mean([t['soft_mix_mse'] for t in task_stats])),
            'macro_uniform_mix_mse':float(np.mean([t['uniform_mix_mse'] for t in task_stats])),
            'macro_mse_ratio_to_uniform':float(np.nanmean([t['soft_mix_mse_ratio_to_uniform'] for t in task_stats]))})
    return task_rows,summaries
