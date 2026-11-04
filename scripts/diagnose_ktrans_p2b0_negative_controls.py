"""P2-B0 read-only SOURCE pseudo-LOTO negative controls and cluster bootstrap.

Run on already exported 3,337 task-tile units; does not fit a model.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from urban_transfer.ktrans_p2a import samples_from_rows
from urban_transfer.ktrans_p2b0 import run_negative_controls,effect_by_task_cue

def save_csv(path,rows):
    if not rows:raise ValueError('No rows: '+str(path))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('runs/ktrans_competence/education_all_outer_train_dev'))
    p.add_argument('--output',type=Path,default=Path('runs/ktrans_competence/p2b0_education_source_dev'))
    p.add_argument('--repeats',type=int,default=199)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--min-group',type=int,default=8)
    a=p.parse_args()
    if a.repeats<1 or a.min_group<1:p.error('Positive repeats/min-group required')
    protocol=json.loads((a.input/'protocol.json').read_text())
    if protocol.get('version')!='ktrans-empirical-competence-v1':
        raise ValueError('Unexpected input protocol')
    if set(protocol.get('inner_partitions',[]))!={'meta_SCUI','meta_UCUI'}:
        raise ValueError('Expected inner source SCUI and UCUI')
    rows=[]
    with (a.input/'competence_rows.jsonl').open() as f:
        for line in f:
            if line.strip():rows.append(json.loads(line))
    if len(rows)!=int(protocol['n_expert_rows']):
        raise ValueError('Expert row count differs from source protocol')
    samples=samples_from_rows(rows,protocol['outer_target_excluded'])
    if {s['task'] for s in samples}!=set(protocol['source_tasks']):
        raise ValueError('Incomplete source pseudo-task coverage')
    observed,directions,repetitions,comparison=run_negative_controls(
        samples,seed=a.seed,repeats=a.repeats,min_group=a.min_group)
    a.output.mkdir(parents=True,exist_ok=True)
    save_csv(a.output/'observed_direction_details.csv',directions)
    save_csv(a.output/'effect_by_task_cue.csv',effect_by_task_cue(directions))
    save_csv(a.output/'negative_control_replicates.csv',repetitions)
    save_csv(a.output/'negative_control_comparison.csv',comparison)
    (a.output/'protocol.json').write_text(json.dumps({
       'version':'ktrans-p2b0-cue-negative-controls-v1',
       'input':str(a.input.resolve()),
       'source_tasks':protocol['source_tasks'],
       'excluded_outer_target':protocol['outer_target_excluded'],
       'n_task_tile_units':len(samples),'n_expert_rows':len(rows),
       'seed':a.seed,'repeats_per_mode':a.repeats,'min_group':a.min_group,
       'threshold_source':'ORIGINAL SCUI task-specific visual_cues median, not recomputed under shuffles/bootstrap',
       'permutation':'exchange whole VLM score+mask evidence blocks within task/split; and separately within task/split/city',
       'cluster_bootstrap':'sample cities with replacement separately within each pseudo-task/split, keeping all their tile records',
       'observed':observed,
       'caveat':'Descriptive POSTHOC DEV only. 96 cue-expert comparisons correlated, null denominators vary, cities may share images; Monte Carlo fractions are not confirmatory p values. No target-task label or model training.'},indent=2))
    print('[p2b0] source samples=',len(samples),
          'observed_evaluable=',observed['n_evaluable'],
          'observed_agreement=',f'{observed["agreement_rate"]:.3f}',flush=True)
    for item in comparison:
        print(f'[p2b0] {item["mode"]:<26} null_mean={item["null_mean_agreement"]:.3f} '
              f'p05={item["null_p05_agreement"]:.3f} p95={item["null_p95_agreement"]:.3f} '
              f'tail_fraction={item["descriptive_tail_fraction_ge_observed"]:.3f} '
              f'valid={item["n_valid_replicates"]}',flush=True)
    print('[p2b0] saved',a.output,flush=True)

if __name__=='__main__':main()
