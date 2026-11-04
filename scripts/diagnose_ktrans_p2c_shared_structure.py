"""P2-C1/C2 read-only source expert capability sharing and provenance audit.

Inputs are existing source-only competence rows and K5-A INNER last.pt
checkpoints. Runs entirely on CPU; performs no training/VLM inference.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.run_ktrans_p2b1_source_lopo import text_descriptors
from urban_transfer.ktrans_p2a import samples_from_rows
from urban_transfer.ktrans_p2c1 import (
    standardize_effects,task_pair_structure,rank_structure,
    task_semantic_association)
from urban_transfer.ktrans_p2c2 import fold_parameter_audit,training_data_audit

def write_csv(path,rows):
    if not rows:raise ValueError('No output rows: '+str(path))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--k5a-protocol',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--knowledge-cache',type=Path,required=True)
    p.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--extra-reuse-cache',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-group',type=int,default=8)
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--max-pixels',type=int,default=262144)
    args=p.parse_args()
    if args.min_group<1:p.error('min-group must be positive')
    evidence=json.loads((args.input/'protocol.json').read_text())
    if evidence.get('version')!='ktrans-empirical-competence-v1':
        raise ValueError('Source competence provenance version mismatch')
    source=tuple(sorted(evidence['source_tasks']))
    outer=evidence['outer_target_excluded']
    if outer in source or set(evidence.get('inner_partitions',()))!={'meta_SCUI','meta_UCUI'}:
        raise ValueError('Outer task leakage or unexpected inner partitions')
    if Path(evidence['k5a_protocol']).resolve()!=args.k5a_protocol.resolve():
        raise ValueError('Wrong K5-A protocol for competence export')
    if Path(evidence['manifest']).resolve()!=args.manifest.resolve():
        raise ValueError('Wrong full source manifest')
    k5a=json.loads(args.k5a_protocol.read_text())
    if k5a['outer_task']!=outer or set(k5a['inner_source_tasks'])!=set(source):
        raise ValueError('Inner source tasks mismatch')
    if Path(k5a['outer_checkpoint']).resolve()!=Path(evidence['k4_checkpoint_for_provenance_only']).resolve():
        raise ValueError('Different outer source provenance checkpoint')
    manifest=json.loads(args.manifest.read_text())
    if manifest['heldout_task']!=outer:
        raise ValueError('Source manifest outer task mismatch')
    texts,keys=text_descriptors(manifest,
        [args.reuse_cache,*args.extra_reuse_cache,args.knowledge_cache],
        args.model,args.max_pixels,source)
    rows=[]
    with (args.input/'competence_rows.jsonl').open() as f:
        for line in f:
            if line.strip():rows.append(json.loads(line))
    if len(rows)!=int(evidence['n_expert_rows']):
        raise ValueError('Source competence expert count mismatch')
    samples=samples_from_rows(rows,outer)
    if {s['task'] for s in samples}!=set(source) or any(
        set(s['experts'])!=set(source)-{s['task']} for s in samples):
        raise ValueError('Incomplete source experts/task folds')
    names,tensor,masks,effects,thresholds,scales=standardize_effects(
        samples,source,args.min_group)
    comparisons=task_pair_structure(names,tensor,masks,texts)
    rank=rank_structure(names,tensor,masks,'meta_SCUI')
    associations=[{'inner_partition':split,**task_semantic_association(
        [r for r in comparisons if r['inner_partition']==split])}
        for split in ('meta_SCUI','meta_UCUI')]
    csv_path=args.k5a_protocol.parent/'inner_source_only.csv'
    if not csv_path.is_file():raise FileNotFoundError(str(csv_path))
    checkpoints={}
    for task in source:
        path=args.k5a_protocol.parent/'inner_models'/task/'last.pt'
        if not path.is_file():raise FileNotFoundError(str(path))
        # Own training checkpoints, read-only model state / metadata on CPU.
        checkpoints[task]=torch.load(path,map_location='cpu',weights_only=False)
    foldmeta,params,shared=fold_parameter_audit(checkpoints,source,outer,csv_path)
    supervised,overlap=training_data_audit(foldmeta,csv_path,source,outer)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'cue_effects.csv',effects)
    write_csv(args.output/'task_pair_ability_similarity.csv',comparisons)
    write_csv(args.output/'masked_pair_svd.csv',rank)
    write_csv(args.output/'text_ability_similarity.csv',associations)
    write_csv(args.output/'checkpoint_fold_metadata.csv',foldmeta)
    write_csv(args.output/'named_expert_parameter_shift.csv',params)
    write_csv(args.output/'shared_trunk_parameter_shift.csv',shared)
    write_csv(args.output/'fold_supervised_source_counts.csv',supervised)
    write_csv(args.output/'fold_training_overlap.csv',overlap)
    (args.output/'protocol.json').write_text(json.dumps({
        'version':'ktrans-p2c1-c2-source-sharing-audit-v1',
        'source_tasks':names,'outer_target_excluded':outer,
        'input':str(args.input.resolve()),
        'k5a_protocol':str(args.k5a_protocol.resolve()),
        'nested_csv':str(csv_path.resolve()),
        'source_manifest':str(args.manifest.resolve()),
        'text_only_source_prompt_cache_keys':keys,
        'n_source_task_tile_units':len(samples),'n_expert_rows':len(rows),
        'min_group':args.min_group,
        'scui_scale_by_task':scales,
        'thresholds_source':'ORIGINAL SCUI per pseudo task ordinal cue medians',
        'missing_data_policy':'Pseudo target own expert absent; leave diagonal NaN/masked; do not impute',
        'shared_expert_comparison':'For pair of pseudo tasks compare only three common named source experts, each six cue effects',
        'low_rank_caveat':'Only pairwise complete 2×N effect blocks; rank1 fraction does not estimate global 5×30 matrix rank or out-of-task transfer',
        'source_parameter_caveat':'Parameter cosines compare named expert heads only, while shared trunk differs; different weights prove not a frozen identical expert function',
        'selection_warning':'POSTHOC DEV; UCUI inspected; K4 best.pt was selected using outer Education metadata, so no independent zero-shot claim'
    },indent=2))
    print('[p2c] source_tiles=',len(samples),'expert_rows=',len(rows),
          'source_tasks=',len(source),flush=True)
    for split in ('meta_SCUI','meta_UCUI'):
        scored=[r for r in comparisons if r['inner_partition']==split
                and np.isfinite(r['competence_effect_cosine'])]
        mean=float(np.mean([r['competence_effect_cosine'] for r in scored])) if scored else float('nan')
        assoc=next(r for r in associations if r['inner_partition']==split)
        print(f'[p2c1] {split} comparable_task_pairs={len(scored)}/{len(source)*(len(source)-1)//2} '
              f'mean_shared_expert_cue_cosine={mean:.3f} '
              f'text_vs_competence_pearson={assoc["pearson"]:.3f}',flush=True)
    valid=[r for r in params if np.isfinite(r['cosine_parameter_similarity'])]
    print('[p2c2] named_expert_parameter_pairs=',len(valid),
          'identical=',sum(r['identical_parameters'] for r in valid),
          'mean_parameter_cosine=',f'{np.mean([r["cosine_parameter_similarity"] for r in valid]):.3f}' if valid else 'N/A',
          'mean_train_row_overlap=',f'{np.mean([r["training_row_jaccard"] for r in overlap]):.3f}',flush=True)
    print('[p2c] saved',args.output,flush=True)

if __name__=='__main__':main()
