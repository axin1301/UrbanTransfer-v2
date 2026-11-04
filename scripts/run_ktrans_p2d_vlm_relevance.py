"""P2-D-VLM: compare source LOPO using actual Qwen3-VL task+image relevance.

Runs both original K5-A folds and P2-C2B stable expert bank if supplied,
with EXACT same task-tile knowledge, SCUI/UCUI split and feature design.
No new VLM calls and no UCUI labels enter training or model selection.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.run_ktrans_p2b1_source_lopo import text_descriptors
from urban_transfer.ktrans_p2b1 import aligned_samples,summarize_source_lopo
from urban_transfer.ktrans_p2d_vlm import (
    run_cross_task,run_same_task,summarize_same_task)

def csv_write(path,records):
    if not records:raise ValueError('Missing result records')
    keys=list(dict.fromkeys(k for row in records for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader();w.writerows(records)

def source_evidence(directory,source,outer,stable):
    if stable:
        protocol=json.loads((directory/'protocol.json').read_text())
        if protocol.get('version')!='ktrans-stable-independent-source-ridge-v1':
            raise ValueError('Expected P2-C2B stable expert evidence version')
        if set(protocol['source_tasks'])!=set(source) or protocol['outer_target_excluded']!=outer:
            raise ValueError('Stable source task/outer target mismatch')
        path=directory/'stable_evidence'/'competence_rows.jsonl'
        size=protocol['n_rows']
        hashes=protocol['fixed_expert_hashes']
    else:
        protocol=json.loads((directory/'protocol.json').read_text())
        if protocol.get('version')!='ktrans-empirical-competence-v1':
            raise ValueError('Expected original source competence evidence')
        if set(protocol['source_tasks'])!=set(source) or protocol['outer_target_excluded']!=outer:
            raise ValueError('Original source task/outer target mismatch')
        path=directory/'competence_rows.jsonl'
        size=protocol['n_expert_rows']
        hashes=None
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows)!=size:raise ValueError('Source evidence count mismatch')
    if hashes is not None and any(row['expert_model_hash']!=hashes[row['expert']]
                                  for row in rows):
        raise ValueError('Expert identity hash changed across pseudo folds')
    return rows

def validate_matched(original,stable):
    """Ensure only predictions/losses have been replaced by stable bank."""
    keys=['pseudo_task','inner_partition','tile_id','city','expert',
          'image_path','knowledge_cache_key','label_log1p',
          'knowledge_scores','knowledge_masks']
    a=sorted(original,key=lambda r:(r['pseudo_task'],r['inner_partition'],r['tile_id'],r['expert']))
    b=sorted(stable,key=lambda r:(r['pseudo_task'],r['inner_partition'],r['tile_id'],r['expert']))
    if len(a)!=len(b):raise ValueError('Different source pseudo task evidence length')
    for left,right in zip(a,b):
        if any(left[k]!=right[k] for k in keys):
            raise ValueError('Different tile/task/VLM source knowledge; cannot compare')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-input',type=Path,default=Path('runs/ktrans_competence/education_all_outer_train_dev'))
    p.add_argument('--stable-input',type=Path,default=Path('runs/ktrans_competence/p2c2b_stable_source_dev'))
    p.add_argument('--manifest',type=Path,default=None)
    p.add_argument('--knowledge-cache',type=Path,default=Path('runs/ktrans_competence/full_source_evidence.jsonl'))
    p.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--extra-reuse-cache',type=Path,action='append',default=[Path('runs/k7d_education/evidence.jsonl')])
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--alpha',type=float,default=10.)
    p.add_argument('--temperature',type=float,default=.5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,default=Path('runs/ktrans_competence/p2d_vlm_source_dev'))
    args=p.parse_args()
    if args.alpha<=0 or args.temperature<=0:p.error('Positive alpha and temperature required')
    original_meta=json.loads((args.original_input/'protocol.json').read_text())
    source=tuple(sorted(original_meta['source_tasks']))
    outer=original_meta['outer_target_excluded']
    if args.manifest is None:args.manifest=Path(original_meta['manifest'])
    if Path(original_meta['manifest']).resolve()!=args.manifest.resolve():
        raise ValueError('Source manifest mismatch')
    manifest=json.loads(args.manifest.read_text())
    if manifest['heldout_task']!=outer or set(manifest['source_tasks'])!=set(source):
        raise ValueError('Outer target or source task mismatch')
    descriptors,cache_keys=text_descriptors(manifest,
        [args.reuse_cache,*args.extra_reuse_cache,args.knowledge_cache],
        args.model,args.max_pixels,source)
    original=source_evidence(args.original_input,source,outer,False)
    stable=source_evidence(args.stable_input,source,outer,True)
    validate_matched(original,stable)
    args.output.mkdir(parents=True,exist_ok=True)
    metadata={}
    for bank,raw in (('original_k5a',original),('stable_ridge',stable)):
        samples=aligned_samples(raw,outer,source)
        if len(samples)*4!=len(raw):raise ValueError('Incomplete paired expert records')
        cross=run_cross_task(samples,source,descriptors,args.alpha,args.temperature,args.seed)
        per_task,macro=summarize_source_lopo(cross)
        same=run_same_task(samples,source,descriptors,args.alpha,args.temperature,args.seed)
        same_task,same_macro=summarize_same_task(same)
        dest=args.output/bank;dest.mkdir(parents=True,exist_ok=True)
        for name,rows in (
            ('cross_task_metrics.csv',per_task),
            ('cross_macro_metrics.csv',macro),
            ('same_task_ucui_metrics.csv',same_task),
            ('same_macro_ucui.csv',same_macro)):
            csv_write(dest/name,rows)
        metadata[bank]={'n_tiles':len(samples),
                        'n_source_expert_rows':len(raw),
                        'n_cross_metric_rows':len(cross)}
        for row in macro:
            if row['inner_partition']!='meta_UCUI':continue
            print(f'[p2d-vlm] {bank:<13} cross_UCUI {row["model"]:<29} '
                  f'pairwise={row["macro_pairwise_accuracy"]:.3f} '
                  f'mse_ratio_uniform={row["macro_mse_ratio_to_uniform"]:.3f}',flush=True)
        for row in same_macro:
            print(f'[p2d-vlm] {bank:<13} same_UCUI  {row["model"]:<29} '
                  f'pairwise={row["macro_pairwise_accuracy"]:.3f} '
                  f'mse_ratio_uniform={row["macro_mse_ratio_to_uniform"]:.3f}',flush=True)
    (args.output/'protocol.json').write_text(json.dumps({
        'version':'ktrans-p2d-vlm-task-conditioned-relevance-v1',
        'original_source_evidence':str(args.original_input.resolve()),
        'stable_source_evidence':str(args.stable_input.resolve()),
        'source_tasks':source,'excluded_outer_target':outer,
        'source_manifest':str(args.manifest.resolve()),
        'text_only_cache_keys':cache_keys,
        'alpha':args.alpha,'temperature':args.temperature,'seed':args.seed,
        'model_variants':[
            'task_expert (original P2-B1 baseline)',
            'task_cue_expert (original P2-B1 baseline)',
            'vlm_task_relevance (direct image_task relevance)',
            'vlm_task_residual (image_task relevance minus text_only prior)',
            'vlm_task_full (direct and residual relevance)',
            'permuted_vlm_task_relevance',
            'permuted_vlm_task_full'],
        'permutation':'shuffle image_task evidence_relevance scores AND masks across source tiles within same task/split, preserve visual_cues, evidence_confidence, experts and labels',
        'cross_task':'train OTHER source pseudo-tasks SCUI only; test heldout pseudo task SCUI/UCUI',
        'same_task':'train current pseudo task SCUI labels; test UCUI, SUPERVISED diagnostic only',
        'unit':'one pseudo-task/tile, four named source experts; equal source task macro aggregation',
        'comparison':'Both banks use IDENTICAL source task/tile/labels/knowledge; experts/scale differ, so do not infer causal bank identity effect.',
        'banks':metadata,
        'warning':'POSTHOC DEV source-only evidence. Repeated inspection of inner UCUI and original K4 best.pt Education selection invalidate independent zero-shot claims. New exploratory feature variants predeclared in code; do not tune against UCUI.'
    },indent=2))
    print('[p2d-vlm] saved',args.output,flush=True)

if __name__=='__main__':main()
