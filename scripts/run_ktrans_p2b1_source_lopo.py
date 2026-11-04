"""P2-B1 CPU: SOURCE pseudo-task leave-one-task-out ridge competence baselines.

Fixed ridge penalty and softmax temperature; NO target label tuning and
no source UCUI model selection. Text-only VLM source task descriptors
are validated against exact canonical cache job identity.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.cache_ktrans_k7d import unique_jobs,merge_records
from urban_transfer.ktrans_knowledge_expert_alignment import cue_relevance
from urban_transfer.ktrans_p2b1 import aligned_samples,source_lopo,summarize_source_lopo

def text_descriptors(manifest,caches,model,max_pixels,tasks):
    if manifest.get('source_manifest_mode')!='outer-train-full-v1':
        raise ValueError('Expected full outer source-TRAIN manifest')
    if set(manifest['source_tasks'])!=set(tasks) or manifest['heldout_task'] in tasks:
        raise ValueError('Manifest source/outer target mismatch')
    if any(r['partition']!='train' or r['task'] not in tasks for r in manifest['rows']):
        raise ValueError('Manifest contains non-source rows')
    records=merge_records(caches)
    jobs={j['task']:j for j in unique_jobs(manifest,model,max_pixels)
          if j['condition']=='text_only'}
    out={}
    keys={}
    for task in tasks:
        j=jobs.get(task)
        if j is None:raise ValueError('Missing source-task conceptual VLM prompt '+task)
        record=records.get(j['key'])
        if not record or record.get('parsed') is None or any(
            record.get(k)!=j[k] for k in ('version','tile_id','task','condition','model','prompt','max_pixels')):
            raise ValueError('Missing/incorrect text-only VLM cache for '+task)
        # No image or target-task measurement is accessed.
        out[task]=cue_relevance(record['parsed']).tolist()
        keys[task]=j['key']
    return out,keys

def write_csv(path,records):
    if not records:raise ValueError('No outputs for '+str(path))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)

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
    args=p.parse_args()
    if args.ridge_alpha<=0 or args.temperature<=0:p.error('alpha/temperature must be positive')
    protocol=json.loads((args.input/'protocol.json').read_text())
    if protocol.get('version')!='ktrans-empirical-competence-v1':
        raise ValueError('Expected empirical source-only competence v1')
    if set(protocol.get('inner_partitions',()))!={'meta_SCUI','meta_UCUI'}:
        raise ValueError('Expected source inner SCUI/UCUI')
    manifest=json.loads(args.manifest.read_text())
    if Path(protocol['manifest']).resolve()!=args.manifest.resolve():
        raise ValueError('Competence source manifest mismatch')
    if Path(protocol['k4_checkpoint_for_provenance_only']).resolve()!=Path(manifest['k4_checkpoint']).resolve():
        raise ValueError('K4 provenance mismatch')
    source=tuple(sorted(protocol['source_tasks']))
    if manifest['heldout_task']!=protocol['outer_target_excluded']:
        raise ValueError('Outer heldout source isolation broken')
    texts,keys=text_descriptors(manifest,
        [args.reuse_cache,*args.extra_reuse_cache,args.knowledge_cache],
        args.model,args.max_pixels,source)
    rows=[]
    with (args.input/'competence_rows.jsonl').open() as f:
        for line in f:
            if line.strip():rows.append(json.loads(line))
    if len(rows)!=protocol['n_expert_rows']:
        raise ValueError('Input row provenance count mismatch')
    samples=aligned_samples(rows,protocol['outer_target_excluded'],source)
    if len(samples)*4!=len(rows) or {s['task'] for s in samples}!=set(source):
        raise ValueError('Incomplete source-only paired expert evidence')
    outputs=source_lopo(samples,source,texts,
                        alpha=args.ridge_alpha,temperature=args.temperature,seed=args.seed)
    task_metrics,macro_metrics=summarize_source_lopo(outputs)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'source_task_metrics.csv',task_metrics)
    write_csv(args.output/'macro_metrics.csv',macro_metrics)
    write_csv(args.output/'tile_predictions.csv',outputs)
    (args.output/'protocol.json').write_text(json.dumps({
        'version':'ktrans-p2b1-source-task-lopo-v1',
        'input':str(args.input.resolve()),'manifest':str(args.manifest.resolve()),
        'excluded_outer_target':protocol['outer_target_excluded'],
        'source_tasks':source,'text_only_cache_keys':keys,
        'ridge_alpha':args.ridge_alpha,'temperature':args.temperature,
        'hyperparameters':'fixed BEFORE accessing P2-B1 results; no UCUI selection',
        'source_crossfit':'for each pseudo task train ridge on OTHER source tasks SCUI only, evaluate heldout source task SCUI & UCUI',
        'target':'per-tile loss centered and divided by within-tile expert-loss range (bounded), not inferenced from labels at prediction',
        'permuted_control':'shuffle whole task-image knowledge blocks separately in source training SCUI and heldout task eval split; task descriptors remain intact',
        'unit':'one pseudo task/tile, four paired expert errors; macro metrics equal weight by pseudo task',
        'n_tile_units':len(samples),'n_prediction_rows':len(outputs),
        'warning':'DEV ONLY: K4 best.pt meta-selected on outer Education. Inner source fold expert identities represent separately retrained models, not outer K4 expert parameters. SCUI and UCUI have previously been inspected. This is exploratory evidence, NOT strict outer zero-shot generalization.'
        },indent=2))
    print('[p2b1] source_tile_units=',len(samples),'output_rows=',len(outputs),flush=True)
    for row in macro_metrics:
        print(f'[p2b1] {row["model"]:<27} {row["inner_partition"]:<9} '
              f'top1={row["macro_top1_accuracy"]:.3f} '
              f'pairwise={row["macro_pairwise_accuracy"]:.3f} '
              f'blend_MSE={row["macro_soft_mix_mse"]:.3f} '
              f'uniform_MSE={row["macro_uniform_mix_mse"]:.3f}',flush=True)
    print('[p2b1] saved',args.output,flush=True)

if __name__=='__main__':main()
