"""P2-E1 posthoc exploratory source LOPO of VLM expert capability hypotheses.

All VLM statements are produced WITHOUT access to expert losses.
Only OTHER source tasks SCUI train a fixed small Ridge calibrator.
Validation on heldout pseudo task UCUI; no UCUI tuning. Pilot n=4/cell
is extremely underpowered: report descriptive results, not claims.
"""
from collections import defaultdict
import numpy as np
from urban_transfer.ktrans_p2b1 import (
    aligned_samples,relative_target,eval_predictions,aggregate)

MODES=('definition_only','definition_relation','definition_relation_region',
       'permuted_relation_region')

def collect_hypotheses(manifest,records):
    jobs={j['key']:j for j in manifest['jobs']}
    output={}
    for r in records:
        if r.get('parsed') is None or r.get('key') not in jobs:continue
        job=jobs[r['key']]
        key=(job['arm'],job['task'],job['tile_id'])
        if key in output:raise ValueError('Duplicate valid hypothesis job')
        output[key]=r['parsed']['experts']
    return output

def feature(expert,source_tasks,hypothesis):
    one=np.zeros(len(source_tasks),dtype=float)
    one[source_tasks.index(expert)]=1.
    # The same named source expert gets the same global coordinate
    # across all folds. Missing scores use mask=0, not invented zero.
    values=[];mask=[]
    for name in ('compatibility','region_support','transfer_risk','uncertainty'):
        v=hypothesis[name]
        values.append(0. if v is None else float(v)/3.)
        mask.append(float(v is not None))
    return np.concatenate(([1.],one,
        np.outer(one,values).reshape(-1),
        np.outer(one,mask).reshape(-1)))

def run_lopo(evidence_rows,manifest,hypotheses,source_tasks,
             alpha=10.,temperature=.5):
    if alpha<=0 or temperature<=0:raise ValueError('Positive fixed settings required')
    tasks=tuple(sorted(source_tasks))
    allowed={(j['task'],j['split'],j['tile_id']) for j in manifest['tiles']}
    filtered=[r for r in evidence_rows
              if (r['pseudo_task'],r['inner_partition'],r['tile_id']) in allowed]
    samples=aligned_samples(filtered,manifest['plan']['outer_target_excluded'],tasks)
    if len(samples)!=len(manifest['tiles']):
        raise ValueError('Incomplete pilot evaluation tile/expert coverage')
    out=[]
    for mode in MODES:
        lookup={}
        for s in samples:
            key=(mode,s['task'],s['tile'] if mode.endswith('_region') else None)
            if key not in hypotheses:raise ValueError('Missing validated VLM hypothesis '+str(key))
            lookup[s['task'],s['split'],s['tile']]=hypotheses[key]
        for target in tasks:
            train=[s for s in samples if s['task']!=target and s['split']=='meta_SCUI']
            test=[s for s in samples if s['task']==target and s['split']=='meta_UCUI']
            if not train or not test:raise ValueError('Empty source heldout fold')
            def encode(group,train_labels):
                x=[];y=[];counts=[]
                for s in group:
                    experts=lookup[s['task'],s['split'],s['tile']]
                    if set(experts)!=set(s['experts']):
                        raise ValueError('VLM source expert identity mismatch')
                    counts.append(len(s['experts']))
                    for e in s['experts']:
                        x.append(feature(e,tasks,experts[e]))
                    if train_labels:y.extend(relative_target(s))
                return np.asarray(x,dtype=float),np.asarray(y,dtype=float),counts
            x,y,_=encode(train,True)
            v,_,counts=encode(test,False)
            penalty=np.ones(x.shape[1]);penalty[0]=0.
            co=np.linalg.solve(x.T@x+alpha*np.diag(penalty),x.T@y)
            flat=v@co;offset=0;pred=[]
            for count in counts:
                pred.append(flat[offset:offset+count]);offset+=count
            results=eval_predictions(test,pred,temperature)
            for row in results:row['mode']=mode;out.append(row)
    by=defaultdict(list)
    for row in out:by[row['mode'],row['pseudo_task']].append(row)
    per_task=[{'mode':mode,'pseudo_task':task,**aggregate(rows)}
              for (mode,task),rows in sorted(by.items())]
    by_mode=defaultdict(list)
    for row in per_task:by_mode[row['mode']].append(row)
    macro=[{'mode':mode,'n_tasks':len(vals),
            'n_tiles':sum(x['n_tiles'] for x in vals),
            'macro_pairwise_accuracy':float(np.mean([x['pairwise_accuracy'] for x in vals])),
            'macro_top1_accuracy':float(np.mean([x['top1_fractional_accuracy'] for x in vals])),
            'macro_mse_ratio_to_uniform':float(np.nanmean(
                [x['soft_mix_mse_ratio_to_uniform'] for x in vals]))}
            for mode,vals in sorted(by_mode.items())]
    return per_task,macro
