"""Evaluate completed P2-E1 pilot with fixed source LOTO; descriptive only."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_ktrans_p2e1_indicator_pilot import cache_records,is_valid
from urban_transfer.ktrans_p2e1_pilot import parse,descriptive_audit
from urban_transfer.ktrans_p2e1_eval import collect_hypotheses,run_lopo
from scripts.run_ktrans_p2b1_source_lopo import text_descriptors
from urban_transfer.ktrans_p2b1 import aligned_samples,source_lopo,summarize_source_lopo

def write_csv(p,rows):
    if not rows:raise ValueError('Empty metric rows')
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,default=Path('runs/ktrans_competence/p2e1_qwen3vl32b_source_dev'))
    p.add_argument('--input',type=Path,default=Path('runs/ktrans_competence/education_all_outer_train_dev'))
    p.add_argument('--knowledge-cache',type=Path,default=Path('runs/ktrans_competence/full_source_evidence.jsonl'))
    p.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--extra-reuse-cache',type=Path,action='append',default=[Path('runs/k7d_education/evidence.jsonl')])
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--alpha',type=float,default=10.)
    p.add_argument('--temperature',type=float,default=.5)
    a=p.parse_args()
    manifest=json.loads((a.pilot/'pilot_manifest.json').read_text())
    plan=manifest['plan'];sources=tuple(sorted(plan['source_tasks']))
    original=json.loads((a.input/'protocol.json').read_text())
    if original['outer_target_excluded']!=plan['outer_target_excluded'] or set(original['source_tasks'])!=set(sources):
        raise ValueError('Source/outer protocol mismatch')
    if str(a.input.resolve())!=plan['source_evidence']:
        raise ValueError('Wrong source evidence input')
    cache=cache_records(a.pilot/'vlm_evidence.jsonl')
    valid=[]
    for job in manifest['jobs']:
        record=cache.get(job['key'])
        if not is_valid(job,record):continue
        parsed=parse(record['raw'],sources,job['task'],job['image_path'] is not None)
        if parsed is not None:valid.append({**record,'parsed':parsed})
    if len(valid)!=len(manifest['jobs']):
        raise ValueError(f'Pilot incomplete: valid={len(valid)}/{len(manifest["jobs"])}; do not evaluate partial tasks')
    evidence=[json.loads(l) for l in (a.input/'competence_rows.jsonl').read_text().splitlines() if l.strip()]
    if len(evidence)!=original['n_expert_rows']:raise ValueError('Wrong paired evidence count')
    hypotheses=collect_hypotheses(manifest,valid)
    task,macro=run_lopo(evidence,manifest,hypotheses,sources,
                        a.alpha,a.temperature)
    # Exact matched pilot baseline: original P2-B1 Task–Expert,
    # trained only on OTHER source pseudo tasks' SCUI.
    evidence_manifest=json.loads(Path(original['manifest']).read_text())
    descriptors,keys=text_descriptors(evidence_manifest,
        [a.reuse_cache,*a.extra_reuse_cache,a.knowledge_cache],
        a.model,a.max_pixels,sources)
    allowed={(j['task'],j['split'],j['tile_id']) for j in manifest['tiles']}
    sampled=[r for r in evidence if
             (r['pseudo_task'],r['inner_partition'],r['tile_id']) in allowed]
    samples=aligned_samples(sampled,original['outer_target_excluded'],sources)
    base=source_lopo(samples,sources,descriptors,a.alpha,a.temperature,seed=plan['seed'])
    base_task,base_macro=summarize_source_lopo(base)
    base_task=[{'mode':'task_expert',**{k:v for k,v in row.items()
                       if k not in ('model','inner_partition')}}
               for row in base_task
               if row['model']=='task_expert' and row['inner_partition']=='meta_UCUI']
    base_macro=[row for row in base_macro
                if row['model']=='task_expert' and row['inner_partition']=='meta_UCUI']
    if len(base_task)!=5 or len(base_macro)!=1:
        raise ValueError('Incomplete matched Task–Expert baseline')
    task+=base_task
    macro+=[{'mode':'task_expert','n_tasks':base_macro[0]['n_source_pseudo_tasks'],
             'n_tiles':base_macro[0]['n_tiles'],
             'macro_pairwise_accuracy':base_macro[0]['macro_pairwise_accuracy'],
             'macro_top1_accuracy':base_macro[0]['macro_top1_accuracy'],
             'macro_mse_ratio_to_uniform':base_macro[0]['macro_mse_ratio_to_uniform']}]
    write_csv(a.pilot/'p2e1_source_lopo_task.csv',task)
    write_csv(a.pilot/'p2e1_source_lopo_macro.csv',macro)
    for row in macro:
        print(f'[p2e1-eval] {row["mode"]:<29} '
              f'pairwise={row["macro_pairwise_accuracy"]:.3f} '
              f'top1={row["macro_top1_accuracy"]:.3f} '
              f'mse_ratio_uniform={row["macro_mse_ratio_to_uniform"]:.3f}',flush=True)
    print('[p2e1-eval] descriptive DEV only; 4 tiles / task-city split, UCUI already previously inspected',flush=True)

if __name__=='__main__':main()
