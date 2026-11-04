"""P2-A: label-based SOURCE pseudo-task expert-competence diagnostics only.

No estimator trained, no VLM inference or outer-task evaluation.
Each unit is one (pseudo_task, inner_partition, tile), holding all
source-expert errors; never treat expert rows as independent samples.
Thresholds for high/low VLM cues are frozen using source SCUI only.
"""
from collections import defaultdict
import numpy as np
from urban_transfer.ktrans_k7d import CUES

SPLITS=('meta_SCUI','meta_UCUI')
FEATURES=('visual_cues','evidence_relevance','evidence_confidence')
N_CUES=len(CUES)

def samples_from_rows(rows,outer_target):
    grouped=defaultdict(list)
    for row in rows:
        task=row['pseudo_task'];split=row['inner_partition']
        if task==outer_target or row['expert']==outer_target or task==row['expert']:
            raise ValueError('Outer target or own pseudo-task expert in source-only records')
        if split not in SPLITS:raise ValueError('Unexpected inner partition')
        if not np.isfinite(float(row['squared_error'])) or float(row['squared_error'])<0:
            raise ValueError('Invalid expert squared error')
        key=(task,split,row['tile_id'])
        grouped[key].append(row)
    if not grouped:raise ValueError('No competence rows')
    samples=[]
    expected_experts={}
    for (task,split,tile),items in sorted(grouped.items()):
        experts=[r['expert'] for r in items]
        if len(experts)!=len(set(experts)) or len(experts)<2:
            raise ValueError('Duplicate or incomplete expert set')
        reference=items[0]
        for item in items[1:]:
            for field in ('city','knowledge_cache_key','label_log1p','image_path'):
                if item[field]!=reference[field]:
                    raise ValueError('Inconsistent sample metadata/label/VLM knowledge')
            if item['knowledge_scores']!=reference['knowledge_scores'] or item['knowledge_masks']!=reference['knowledge_masks']:
                raise ValueError('Inconsistent VLM feature on same sample')
        names=tuple(sorted(experts))
        if task in expected_experts and expected_experts[task]!=names:
            raise ValueError('Inconsistent source experts across samples/splits')
        expected_experts[task]=names
        a=np.asarray(reference['knowledge_scores'],dtype=float)
        m=np.asarray(reference['knowledge_masks'],dtype=float)
        if a.shape!=(3*N_CUES,) or m.shape!=(3*N_CUES,) or not np.isfinite(a).all() or not np.isfinite(m).all():
            raise ValueError('Invalid VLM score/mask dimension')
        if ((a<0)|(a>1)).any() or not np.isin(m,[0.,1.]).all():
            raise ValueError('Invalid ordinal/mask values')
        # Each pair has the same source-task scaling, so compare within tile.
        loss=np.array([next(float(r['squared_error']) for r in items if r['expert']==n) for n in names])
        if not np.isfinite(loss).all():raise ValueError('Nonfinite losses')
        samples.append({'task':task,'split':split,'tile':tile,'city':reference['city'],
            'experts':names,'loss':loss,'relative_loss':loss-loss.mean(),
            'visual':a.reshape(3,N_CUES),'mask':m.reshape(3,N_CUES)})
    return samples

def task_cue_thresholds(samples):
    """Use SCUI distribution, never UCUI, to choose mid-rank split cutoffs."""
    values=defaultdict(list)
    for sample in samples:
        if sample['split']!='meta_SCUI':continue
        for j,cue in enumerate(CUES):
            # Visually observed ordinal score only (0,1/3,2/3,1).
            if sample['mask'][0,j]:
                values[(sample['task'],cue)].append(float(sample['visual'][0,j]))
    return {(task,cue):float(np.median(vals)) for (task,cue),vals in values.items()}

def cue_group(sample,cue,thresholds):
    j=CUES.index(cue)
    key=(sample['task'],cue)
    if key not in thresholds or sample['mask'][0,j]==0:return 'missing'
    v=float(sample['visual'][0,j])
    # Discrete scores can tie at the SCUI median; ignore ties rather
    # than assign them selectively to create artificial separation.
    if v<thresholds[key]-1e-9:return 'low'
    if v>thresholds[key]+1e-9:return 'high'
    return 'tied'

def summary(samples):
    """Task+split expert rankings; n is independent tile/task count."""
    by=defaultdict(list)
    for s in samples:
        by[(s['task'],s['split'])].append(s)
    records=[]
    for (task,split),group in sorted(by.items()):
        experts=group[0]['experts']
        losses=np.stack([s['loss'] for s in group])
        means=losses.mean(0)
        ranks=np.argsort(means)
        for j,expert in enumerate(experts):
            # Fractional tie wins: ties share the minimum's credit.
            min_loss=losses.min(1)
            winners=np.isclose(losses,min_loss,rtol=1e-10,atol=1e-12)
            win=float(np.mean(winners[:,j]/winners.sum(1)))
            records.append({'pseudo_task':task,'inner_partition':split,
                'expert':expert,'n_tiles':len(group),
                'n_cities':len({s['city'] for s in group}),
                'mean_squared_error':float(means[j]),
                'mean_within_tile_relative_loss':float((losses-losses.mean(1,keepdims=True))[:,j].mean()),
                'fractional_top1_win_rate':win,
                'aggregate_mse_rank':int(np.where(ranks==j)[0][0])+1})
    return records

def condition_rows(samples,thresholds,min_group=8):
    """Within task/split, high-vs-low cue conditional relative losses.

    A cue-specific expert contrast is high mean relative error minus
    low mean relative error, not a causal effect or a fitted predictive
    feature importance. Insufficient groups remain explicitly flagged.
    """
    if min_group<1:raise ValueError('min_group must be positive')
    by=defaultdict(list)
    for s in samples:
        for cue in CUES:
            by[(s['task'],s['split'],cue,cue_group(s,cue,thresholds))].append(s)
    results=[]
    for task in sorted({s['task'] for s in samples}):
        for split in SPLITS:
            group=next((v for k,v in by.items() if k[0]==task and k[1]==split),None)
            if group is None:continue
            experts=group[0]['experts']
            for cue in CUES:
                low=by.get((task,split,cue,'low'),[])
                high=by.get((task,split,cue,'high'),[])
                tied=by.get((task,split,cue,'tied'),[])
                missing=by.get((task,split,cue,'missing'),[])
                adequate=len(low)>=min_group and len(high)>=min_group
                for j,expert in enumerate(experts):
                    low_val=float(np.mean([s['relative_loss'][j] for s in low])) if low else float('nan')
                    high_val=float(np.mean([s['relative_loss'][j] for s in high])) if high else float('nan')
                    results.append({'pseudo_task':task,'inner_partition':split,
                        'cue':cue,'expert':expert,'scui_median_threshold':thresholds.get((task,cue),float('nan')),
                        'n_low':len(low),'n_high':len(high),'n_tied':len(tied),
                        'n_missing':len(missing),'enough_samples':adequate,
                        'mean_relative_loss_low':low_val,'mean_relative_loss_high':high_val,
                        'high_minus_low_relative_loss':high_val-low_val if adequate else float('nan')})
    return results

def city_rows(samples):
    by=defaultdict(list)
    for s in samples:by[(s['task'],s['split'],s['city'])].append(s)
    results=[]
    for (task,split,city),group in sorted(by.items()):
        for j,expert in enumerate(group[0]['experts']):
            results.append({'pseudo_task':task,'inner_partition':split,'city':city,
                'expert':expert,'n_tiles':len(group),
                'mean_relative_loss':float(np.mean([s['relative_loss'][j] for s in group]))})
    return results

def cross_city_consistency(condition_records,task_records):
    """Matched SCUI/UCUI direction signs, no hypothesis test on tiles."""
    keyed={(r['pseudo_task'],r['inner_partition'],r['cue'],r['expert']):r
           for r in condition_records}
    result=[]
    for r in condition_records:
        if r['inner_partition']!='meta_SCUI':continue
        key=(r['pseudo_task'],'meta_UCUI',r['cue'],r['expert'])
        u=keyed.get(key)
        if u is None:continue
        a=float(r['high_minus_low_relative_loss'])
        b=float(u['high_minus_low_relative_loss'])
        valid=np.isfinite(a) and np.isfinite(b)
        result.append({'pseudo_task':r['pseudo_task'],'cue':r['cue'],
            'expert':r['expert'],'scui_contrast':a,'ucui_contrast':b,
            'scui_n_low':r['n_low'],'scui_n_high':r['n_high'],
            'ucui_n_low':u['n_low'],'ucui_n_high':u['n_high'],
            'evaluable':bool(valid),
            'same_direction':bool(a*b>0) if valid and a!=0 and b!=0 else None})
    return result

def pairwise_rows(samples):
    """Within-tile strict pairwise expert wins, ties excluded."""
    by=defaultdict(list)
    for s in samples:
        n=len(s['experts'])
        for i in range(n):
            for j in range(i+1,n):
                d=float(s['loss'][i]-s['loss'][j])
                by[(s['task'],s['split'],s['experts'][i],s['experts'][j])].append(d)
    result=[]
    for (task,split,a,b),diffs in sorted(by.items()):
        d=np.asarray(diffs)
        tie=np.isclose(d,0,rtol=1e-10,atol=1e-12)
        effective=~tie
        result.append({'pseudo_task':task,'inner_partition':split,
            'expert_a':a,'expert_b':b,'n_tiles':len(d),
            'n_non_tied':int(effective.sum()),'n_ties':int(tie.sum()),
            'expert_a_win_rate_excluding_ties':float((d[effective]<0).mean()) if effective.any() else float('nan'),
            'mean_paired_squared_error_diff_a_minus_b':float(d.mean())})
    return result
