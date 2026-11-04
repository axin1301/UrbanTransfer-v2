"""P2-B0 SOURCE-only exploratory negative controls; no model training."""
from collections import defaultdict
import numpy as np
from urban_transfer.ktrans_p2a import task_cue_thresholds, condition_rows, cross_city_consistency

def cue_permutation(samples,rng,within_city=False):
    """Shuffle entire VLM scores/masks inside task/split (optionally city).

    Never alter expert losses, label, expert identity or city.
    """
    strata=defaultdict(list)
    for i,s in enumerate(samples):
        key=(s['task'],s['split'],s['city']) if within_city else (s['task'],s['split'])
        strata[key].append(i)
    result=list(samples);moved=0
    for key in sorted(strata):
        ids=strata[key]
        order=rng.permutation(len(ids))
        for index,donor_idx in enumerate(order):
            a=ids[index];donor=ids[int(donor_idx)]
            moved+=a!=donor
            result[a]={**samples[a],'visual':samples[donor]['visual'],
                       'mask':samples[donor]['mask']}
    return result,{'n_evidence_reassigned':moved,
                   'n_singleton_strata':sum(len(v)==1 for v in strata.values()),
                   'n_strata':len(strata)}

def city_bootstrap(samples,rng):
    """Resample city clusters separately for every source task and split."""
    strata=defaultdict(lambda:defaultdict(list))
    for s in samples:strata[(s['task'],s['split'])][s['city']].append(s)
    out=[]
    for key in sorted(strata):
        names=sorted(strata[key])
        for i in rng.choice(len(names),len(names),replace=True):
            out.extend(strata[key][names[int(i)]])
    return out

def consistency_report(samples,thresholds,min_group):
    directions=cross_city_consistency(condition_rows(samples,thresholds,min_group),())
    valid=[r for r in directions if r['evaluable'] and r['same_direction'] is not None]
    num=sum(r['same_direction'] for r in valid)
    return {'n_possible':len(directions),'n_evaluable':len(valid),
            'n_same_sign':int(num),
            'agreement_rate':float(num/len(valid)) if valid else float('nan'),
            'mean_abs_scui_contrast':float(np.mean([abs(r['scui_contrast']) for r in valid])) if valid else float('nan'),
            'mean_abs_ucui_contrast':float(np.mean([abs(r['ucui_contrast']) for r in valid])) if valid else float('nan'),
            'mean_signed_contrast_product':float(np.mean([r['scui_contrast']*r['ucui_contrast'] for r in valid])) if valid else float('nan')},directions

def run_negative_controls(samples,seed=42,repeats=199,min_group=8):
    if repeats<1 or min_group<1 or not samples:raise ValueError('Invalid controls')
    thresholds=task_cue_thresholds(samples)
    observed,directions=consistency_report(samples,thresholds,min_group)
    if observed['n_evaluable']==0:raise ValueError('No evaluable observed comparisons')
    rng=np.random.default_rng(seed)
    repetitions=[]
    modes=('within_task_split','within_task_split_city','city_cluster_bootstrap')
    for mode in modes:
        for rep in range(repeats):
            if mode=='city_cluster_bootstrap':
                alt=city_bootstrap(samples,rng)
                stats={'n_evidence_reassigned':0,'n_singleton_strata':0,'n_strata':0}
            else:
                alt,stats=cue_permutation(samples,rng,mode=='within_task_split_city')
            score,_=consistency_report(alt,thresholds,min_group)
            repetitions.append({'mode':mode,'repeat':rep+1,'seed':seed,**score,**stats})
    comparisons=[]
    for mode in modes:
        null=[r['agreement_rate'] for r in repetitions if r['mode']==mode
              and np.isfinite(r['agreement_rate'])]
        ge=sum(v>=observed['agreement_rate'] for v in null)
        comparisons.append({'mode':mode,'observed_rate':observed['agreement_rate'],
            'observed_evaluable':observed['n_evaluable'],
            'n_valid_replicates':len(null),
            'null_mean_agreement':float(np.mean(null)) if null else float('nan'),
            'null_p05_agreement':float(np.percentile(null,5)) if null else float('nan'),
            'null_p95_agreement':float(np.percentile(null,95)) if null else float('nan'),
            'descriptive_tail_fraction_ge_observed':float((ge+1)/(len(null)+1)),
            'warning':'Posthoc DEV, dependent cues/experts, potentially varying denominators; NOT confirmatory p-value'})
    return observed,directions,repetitions,comparisons

def effect_by_task_cue(directions):
    groups=defaultdict(list)
    for r in directions:
        if r['evaluable'] and r['same_direction'] is not None:
            groups[(r['pseudo_task'],r['cue'])].append(r)
    out=[]
    for (task,cue),rows in sorted(groups.items()):
        out.append({'pseudo_task':task,'cue':cue,
            'n_experts_evaluable':len(rows),
            'same_sign_experts':sum(r['same_direction'] for r in rows),
            'agreement_rate':float(np.mean([r['same_direction'] for r in rows])),
            'mean_abs_scui_contrast':float(np.mean([abs(r['scui_contrast']) for r in rows])),
            'mean_abs_ucui_contrast':float(np.mean([abs(r['ucui_contrast']) for r in rows])),
            'mean_signed_contrast_product':float(np.mean([r['scui_contrast']*r['ucui_contrast'] for r in rows]))})
    return out
