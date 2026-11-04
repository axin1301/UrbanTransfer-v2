"""P2-A: read-only SOURCE pseudo-task competence audit, CPU only.

Uses exported inner LOTO source task labels for descriptive diagnostics;
does NOT train an expert selector or choose a model using UCUI labels.
"""
import argparse,csv,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from urban_transfer.ktrans_p2a import (
    samples_from_rows,task_cue_thresholds,summary,condition_rows,
    city_rows,cross_city_consistency,pairwise_rows)

def write_csv(path,rows):
    if not rows:raise ValueError('No rows to write: '+str(path))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('runs/ktrans_competence/education_all_outer_train_dev'))
    p.add_argument('--output',type=Path,default=Path('runs/ktrans_competence/p2a_education_source_dev'))
    p.add_argument('--min-group',type=int,default=8)
    args=p.parse_args()
    if args.min_group<1:p.error('--min-group must be positive')
    meta=json.loads((args.input/'protocol.json').read_text())
    if meta.get('version')!='ktrans-empirical-competence-v1':
        raise ValueError('Wrong competence input protocol')
    if set(meta.get('inner_partitions',[]))!={'meta_SCUI','meta_UCUI'}:
        raise ValueError('Expected source-only inner SCUI/UCUI')
    rows=[]
    with (args.input/'competence_rows.jsonl').open() as f:
        for line in f:
            if line.strip():rows.append(json.loads(line))
    if len(rows)!=int(meta['n_expert_rows']):
        raise ValueError('Competence row count does not match protocol provenance')
    samples=samples_from_rows(rows,meta['outer_target_excluded'])
    source=set(meta['source_tasks'])
    if {s['task'] for s in samples}!=source:
        raise ValueError('Source pseudo-task coverage incomplete')
    for task in source:
        for split in ('meta_SCUI','meta_UCUI'):
            if not any(s['task']==task and s['split']==split for s in samples):
                raise ValueError('Missing source pseudo-task / inner split: '+task+' '+split)
    thresholds=task_cue_thresholds(samples)
    overview=summary(samples)
    conditional=condition_rows(samples,thresholds,args.min_group)
    paired=pairwise_rows(samples)
    cities=city_rows(samples)
    consistency=cross_city_consistency(conditional,overview)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'task_expert_ranking.csv',overview)
    write_csv(args.output/'cue_conditional_contrasts.csv',conditional)
    write_csv(args.output/'city_expert_profiles.csv',cities)
    write_csv(args.output/'pairwise_wins.csv',paired)
    write_csv(args.output/'scui_ucui_consistency.csv',consistency)
    counts=Counter((s['task'],s['split']) for s in samples)
    (args.output/'protocol.json').write_text(json.dumps({
       'version':'ktrans-p2a-source-only-descriptive-v1',
       'input':str(args.input.resolve()),
       'input_protocol_version':meta['version'],
       'outer_target_excluded':meta['outer_target_excluded'],
       'number_of_independent_tile_task_units':len(samples),
       'number_of_expert_records':len(rows),
       'sample_counts':{f'{t}/{p}':v for (t,p),v in sorted(counts.items())},
       'cue_threshold_policy':'per source pseudo-task SCUI median of OBSERVED visual score; strict high/low and exclude median ties',
       'missing_cue_policy':'null/mask=0 separate, not low',
       'min_group':args.min_group,
       'metric_unit':'one pseudo-task/tile, with four paired source experts',
       'condition_effect_interpretation':'descriptive high-minus-low within-tile-relative squared-error contrast, NOT causal or deployable',
       'uncertainty_warning':'tiles share cities; city summaries supplied; no independent inferential confidence intervals',
       'source_validation_warning':'SCUI already used in K5-A ridge training; UCUI may be inspected for descriptive generalization, not tune and retest',
       'outer_validation_warning':'K4 best.pt meta-selected DEV; source-only diagnostics do not certify final independent zero-shot evaluation'
       },indent=2))
    print('[p2a] samples=',len(samples),'expert_rows=',len(rows),
          'tasks=',len(source),'cities=',len(set((s['task'],s['city']) for s in samples)),flush=True)
    for (task,split),n in sorted(counts.items()):
        ranking=sorted((r for r in overview if r['pseudo_task']==task and r['inner_partition']==split),
                       key=lambda r:r['mean_squared_error'])
        top=ranking[0]
        print(f'[p2a] {task:<31} {split:<9} n={n:<4} '
              f'mean-MSE-best={top["expert"]} MSE={top["mean_squared_error"]:.3f}',flush=True)
    evaluable=[r for r in consistency if r['evaluable']]
    same=[r for r in evaluable if r['same_direction']]
    print(f'[p2a] cue experts evaluable on BOTH splits={len(evaluable)}/{len(consistency)} '
          f'same_sign={len(same)} (descriptive only)',flush=True)
    print('[p2a] saved',args.output,flush=True)

if __name__=='__main__':main()
