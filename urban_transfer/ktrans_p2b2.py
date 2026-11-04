"""P2-B2: diagnose why VLM cue association fails to predict source expert transfer.

CPU only. Same-task SCUI->UCUI is a SUPERVISED diagnostic (not zero-shot).
Cross-task source-LOPO fits only OTHER source pseudo-task SCUI. Both
evaluate IDENTICAL heldout pseudo-task UCUI tiles; no UCUI label fits.
"""
from collections import defaultdict
import numpy as np
from urban_transfer.ktrans_p2b1 import (
    VARIANTS,aggregate,eval_predictions,ridge_predict,source_lopo,soft_route,
    summarize_source_lopo)
from urban_transfer.ktrans_p2b0 import cue_permutation

ROUTES=('same_task_scui_to_ucui','other_tasks_scui_to_ucui')

def source_partition_checks(samples,source_tasks):
    """Check all source pseudo-tasks have separate, complete inner city splits."""
    names=set(source_tasks)
    if len(names)!=len(source_tasks):raise ValueError('Duplicated source tasks')
    for task in names:
        train=[s for s in samples if s['task']==task and s['split']=='meta_SCUI']
        test=[s for s in samples if s['task']==task and s['split']=='meta_UCUI']
        if not train or not test:raise ValueError('Missing SCUI/UCUI source pseudo task '+task)
        if {s['city'] for s in train}&{s['city'] for s in test}:
            raise ValueError('Inner SCUI/UCUI city isolation violated for '+task)
        expected=names-{task}
        if any(set(s['experts'])!=expected for s in train+test):
            raise ValueError('Source expert identity mismatch')
    if any(s['task'] not in names or s['split'] not in ('meta_SCUI','meta_UCUI') for s in samples):
        raise ValueError('Non-source pseudo task or partition')

def same_task_predictions(samples,source_tasks,task_text,alpha=10.,temperature=.5,seed=42):
    """Fit only the SAME pseudo-task SCUI, predict its UCUI.

    This has in-task supervision and is NOT a zero-shot result.
    In particular, the selected fixed expert baseline has also
    seen the pseudo-task SCUI labels.
    """
    source_partition_checks(samples,source_tasks)
    tasks=tuple(sorted(source_tasks))
    out=[]
    for task_idx,task in enumerate(tasks):
        train=[s for s in samples if s['task']==task and s['split']=='meta_SCUI']
        test=[s for s in samples if s['task']==task and s['split']=='meta_UCUI']
        for variant in VARIANTS:
            if variant=='permuted_task_cue_expert':
                rng=np.random.default_rng(seed+1000+task_idx)
                fit,_=cue_permutation(train,rng)
                predict,_=cue_permutation(test,rng)
            else:
                fit,predict=train,test
            scores=ridge_predict(fit,predict,tasks,task_text,variant,alpha)
            for row in eval_predictions(test,scores,temperature):
                row['model']=variant
                row['route']='same_task_scui_to_ucui'
                out.append(row)
    return out

def cross_task_predictions(samples,source_tasks,task_text,alpha=10.,temperature=.5,seed=42):
    source_partition_checks(samples,source_tasks)
    rows=source_lopo(samples,source_tasks,task_text,alpha,temperature,seed)
    return [{**row,'route':'other_tasks_scui_to_ucui'}
            for row in rows if row['inner_partition']=='meta_UCUI']

def route_metrics(rows):
    by=defaultdict(list)
    for row in rows:
        if row['inner_partition']!='meta_UCUI':
            raise ValueError('Only UCUI metrics allowed in route comparison')
        by[(row['route'],row['model'],row['pseudo_task'])].append(row)
    records=[]
    for (route,model,task),tile_rows in sorted(by.items()):
        records.append({'route':route,'model':model,'pseudo_task':task,
                        **aggregate(tile_rows)})
    macro=defaultdict(list)
    for r in records:macro[(r['route'],r['model'])].append(r)
    summary=[]
    for (route,model),vals in sorted(macro.items()):
        summary.append({'route':route,'model':model,
            'n_source_tasks':len(vals),'n_tiles':sum(v['n_tiles'] for v in vals),
            'macro_top1_accuracy':float(np.mean([v['top1_fractional_accuracy'] for v in vals])),
            'macro_pairwise_accuracy':float(np.mean([v['pairwise_accuracy'] for v in vals])),
            'macro_mse_ratio_to_uniform':float(np.nanmean([v['soft_mix_mse_ratio_to_uniform'] for v in vals])),
            'macro_soft_mix_mse':float(np.mean([v['soft_mix_mse'] for v in vals])),
            'macro_uniform_mix_mse':float(np.mean([v['uniform_mix_mse'] for v in vals]))})
    # Compare the exact same tile keys for each task and route.
    keys=defaultdict(set)
    for row in rows:
        k=(row['pseudo_task'],row['city'],row['tile_id'])
        key=(row['model'],row['route'])
        if k in keys[key]:raise ValueError('Duplicate tile evaluation row')
        keys[key].add(k)
    for model in VARIANTS:
        a=keys[model,'same_task_scui_to_ucui']
        b=keys[model,'other_tasks_scui_to_ucui']
        if not a or a!=b:raise ValueError('Cross-task vs same-task UCUI tile mismatch')
    return records,summary

def expert_structure(samples,source_tasks,near_tie_fraction=.05):
    """MSE-based expert ranking diagnostics and signed-residual covariance.

    The pointwise oracle is hindsight-only. Compare against a task's
    SCUI-trained fixed-best expert; do not fit any baseline on UCUI.
    All summaries stratified by pseudo task and partition.
    """
    if not 0<=near_tie_fraction<=1:raise ValueError('Invalid near tie fraction')
    source_partition_checks(samples,source_tasks)
    all_task=[];pairs=[]
    for task in sorted(source_tasks):
        train=[s for s in samples if s['task']==task and s['split']=='meta_SCUI']
        train_mse=np.stack([s['loss'] for s in train]).mean(0)
        fixed_expert=train[0]['experts'][int(np.argmin(train_mse))]
        for split in ('meta_SCUI','meta_UCUI'):
            group=[s for s in samples if s['task']==task and s['split']==split]
            errors=np.stack([s['pred']-s['target'] for s in group])
            loss=np.square(errors)
            means=loss.mean(0)
            rankings=np.argsort(means)
            # Compare individual expert win with group mean MSE rank;
            # stable rank requires train and test same named expert.
            gaps=np.sort(loss,axis=1)[:,1]-np.sort(loss,axis=1)[:,0]
            spans=loss.max(1)-loss.min(1)
            informative=spans>1e-12
            near=(gaps[informative]/spans[informative]<=near_tie_fraction)
            fixed_index=group[0]['experts'].index(fixed_expert)
            uniform=np.square(errors.mean(1))
            oracle=loss.min(1)
            maxspan=np.maximum(spans,1e-12)
            all_task.append({'pseudo_task':task,'inner_partition':split,
                'n_tiles':len(group),'n_cities':len({s['city'] for s in group}),
                'scui_fixed_best_expert':fixed_expert,
                'split_aggregate_best_expert':group[0]['experts'][int(rankings[0])],
                'scui_fixed_best_mse':float(loss[:,fixed_index].mean()),
                'oracle_single_expert_mse_hindsight':float(oracle.mean()),
                'uniform_mix_mse':float(uniform.mean()),
                'fraction_ambiguous_near_tie':float(np.mean(near)) if len(near) else float('nan'),
                'n_informative_tile_spans':int(informative.sum()),
                'median_best_second_gap_over_span':float(np.median(gaps[informative]/maxspan[informative])) if informative.any() else float('nan'),
                'source_expert_mean_mse_spread':float(means.max()-means.min())})
            experts=group[0]['experts']
            for i,a in enumerate(experts):
                for j,b in enumerate(experts):
                    if j<=i:continue
                    x=errors[:,i];y=errors[:,j]
                    stdx=float(np.std(x));stdy=float(np.std(y))
                    cov=float(np.mean((x-x.mean())*(y-y.mean())))
                    corr=float(cov/(stdx*stdy)) if stdx>1e-12 and stdy>1e-12 else float('nan')
                    # Positive residual products are informative for
                    # lack of combination diversity, even with offsets.
                    pairs.append({'pseudo_task':task,'inner_partition':split,
                        'expert_a':a,'expert_b':b,'n_tiles':len(group),
                        'residual_covariance':cov,'residual_pearson':corr,
                        'mean_error_product':float(np.mean(x*y)),
                        'expert_a_mse':float(np.mean(x*x)),
                        'expert_b_mse':float(np.mean(y*y))})
    stability=[]
    by={(r['pseudo_task'],r['inner_partition']):r for r in all_task}
    for task in sorted(source_tasks):
        a=by[task,'meta_SCUI'];b=by[task,'meta_UCUI']
        stability.append({'pseudo_task':task,
            'scui_aggregate_best_expert':a['split_aggregate_best_expert'],
            'ucui_aggregate_best_expert':b['split_aggregate_best_expert'],
            'same_aggregate_best':a['split_aggregate_best_expert']==b['split_aggregate_best_expert'],
            'ucui_scui_fixed_best_mse':b['scui_fixed_best_mse'],
            'ucui_oracle_single_expert_mse_hindsight':b['oracle_single_expert_mse_hindsight'],
            'ucui_uniform_mix_mse':b['uniform_mix_mse']})
    return all_task,pairs,stability
