"""P2-B2 source-only CPU diagnostic: same-task SCUI→UCUI versus source LOTO.

No new VLM calls, no outer Education labels, no target task model fitting.
This is an exploratory developmental analysis of known inner UCUI data.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_ktrans_p2b1_source_lopo import text_descriptors
from urban_transfer.ktrans_p2b1 import aligned_samples
from urban_transfer.ktrans_p2b2 import (
    source_partition_checks,same_task_predictions,cross_task_predictions,
    route_metrics,expert_structure)

def write_csv(path,rows):
    if not rows:raise ValueError('No rows for '+str(path))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--knowledge-cache',required=True,type=Path)
    p.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--extra-reuse-cache',type=Path,action='append',default=[])
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--ridge-alpha',type=float,default=10.)
    p.add_argument('--temperature',type=float,default=.5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--near-tie-fraction',type=float,default=.05)
    a=p.parse_args()
    if a.ridge_alpha<=0 or a.temperature<=0 or not 0<=a.near_tie_fraction<=1:
        p.error('Invalid ridge/temperature/near-tie settings')
    proto=json.loads((a.input/'protocol.json').read_text())
    if proto.get('version')!='ktrans-empirical-competence-v1':
        raise ValueError('Expected empirical source competence input')
    if set(proto.get('inner_partitions',()))!={'meta_SCUI','meta_UCUI'}:
        raise ValueError('Expected source SCUI and UCUI partitions')
    manifest=json.loads(a.manifest.read_text())
    if Path(proto['manifest']).resolve()!=a.manifest.resolve():
        raise ValueError('Competence source manifest mismatch')
    if Path(proto['k4_checkpoint_for_provenance_only']).resolve()!=Path(manifest['k4_checkpoint']).resolve():
        raise ValueError('Source outer K4 checkpoint mismatch')
    if manifest['heldout_task']!=proto['outer_target_excluded']:
        raise ValueError('Outer target mismatch')
    names=tuple(sorted(proto['source_tasks']))
    tasks,keys=text_descriptors(manifest,
        [a.reuse_cache,*a.extra_reuse_cache,a.knowledge_cache],
        a.model,a.max_pixels,names)
    raw=[]
    with (a.input/'competence_rows.jsonl').open() as f:
        for line in f:
            if line.strip():raw.append(json.loads(line))
    if len(raw)!=int(proto['n_expert_rows']):
        raise ValueError('Source expert record count mismatch')
    samples=aligned_samples(raw,proto['outer_target_excluded'],names)
    source_partition_checks(samples,names)
    same=same_task_predictions(samples,names,tasks,a.ridge_alpha,a.temperature,a.seed)
    other=cross_task_predictions(samples,names,tasks,a.ridge_alpha,a.temperature,a.seed)
    all_metrics=same+other
    by_task,macro=route_metrics(all_metrics)
    structure,pairs,rankings=expert_structure(samples,names,a.near_tie_fraction)
    a.output.mkdir(parents=True,exist_ok=True)
    write_csv(a.output/'same_vs_cross_task_ucui.csv',by_task)
    write_csv(a.output/'same_vs_cross_macro.csv',macro)
    write_csv(a.output/'per_tile_ucui_metrics.csv',all_metrics)
    write_csv(a.output/'expert_structure.csv',structure)
    write_csv(a.output/'expert_residual_covariance.csv',pairs)
    write_csv(a.output/'expert_rank_stability.csv',rankings)
    (a.output/'protocol.json').write_text(json.dumps({
        'version':'ktrans-p2b2-failure-diagnosis-v1',
        'input':str(a.input.resolve()),'manifest':str(a.manifest.resolve()),
        'outer_target_excluded':proto['outer_target_excluded'],
        'text_only_source_prompt_cache_keys':keys,
        'source_tasks':names,'n_tile_units':len(samples),
        'n_ucui_tiles':len([s for s in samples if s['split']=='meta_UCUI']),
        'ridge_alpha':a.ridge_alpha,'temperature':a.temperature,
        'seed':a.seed,'near_tie_fraction':a.near_tie_fraction,
        'same_task_protocol':'fit current pseudo task SCUI only; evaluate same task UCUI: SUPERVISED diagnostic, not zero shot',
        'cross_task_protocol':'fit other four pseudo tasks SCUI only; evaluate heldout pseudo task UCUI',
        'comparison':'exact same heldout pseudo task UCUI tile identities, four paired expert predictions',
        'residual_covariance':'signed prediction error after source-only semantic scale; covariance and Pearson per task/split, unpooled',
        'fixed_expert':'choose minimal aggregate SCUI expert MSE for same pseudo task and evaluate frozen expert on UCUI',
        'oracle':'UCUI pointwise best expert is label-dependent hindsight only',
        'warning':'POSTHOC DEV: inner UCUI inspected previously, K4 best.pt outer meta selected; same task uses pseudo task SCUI labels; cross-fold experts separately retrained. Not independent zero-shot test.'
    },indent=2))
    print('[p2b2] source_tiles=',len(samples),'ucui_tiles=',sum(s['split']=='meta_UCUI' for s in samples),flush=True)
    for row in macro:
        print(f'[p2b2] {row["route"]:<29} {row["model"]:<27} '
              f'top1={row["macro_top1_accuracy"]:.3f} '
              f'pairwise={row["macro_pairwise_accuracy"]:.3f} '
              f'mse_ratio_uniform={row["macro_mse_ratio_to_uniform"]:.3f}',flush=True)
    stable=sum(x['same_aggregate_best'] for x in rankings)
    print(f'[p2b2] aggregate expert ranking stability={stable}/{len(rankings)}',flush=True)
    print('[p2b2] saved',a.output,flush=True)

if __name__=='__main__':main()
