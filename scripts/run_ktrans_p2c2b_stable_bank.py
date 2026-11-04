"""P2-C2B end-to-end SOURCE-only stable expert bank + diagnostics.

Reads original full source train, frozen RemoteCLIP features, validated
VLM evidence. Trains independent source-only Ridge experts ONCE with
common inner seen-city partitions. Reuses EXACT same 3337 source
pseudo-task evaluation tiles from K5-A records, but NEVER their old
predictions as training data. CPU only; no VLM calls or GPU training.

Runs P2-B1/B2 and P2-C1 diagnostics immediately on stable evidence.
All findings are exploratory DEV; outer K4 best.pt meta-selected.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from urban_transfer.m3_features import FeatureStore,CachedFeatureDataset
from scripts.run_ktrans_p2b1_source_lopo import text_descriptors
from scripts.diagnose_ktrans_p2b2_failures import write_csv
from urban_transfer.ktrans_p2b1 import aligned_samples,source_lopo,summarize_source_lopo
from urban_transfer.ktrans_p2b2 import (
    same_task_predictions,route_metrics,expert_structure,source_partition_checks)
from urban_transfer.ktrans_p2c1 import (
    standardize_effects,task_pair_structure,rank_structure,task_semantic_association)
from urban_transfer.ktrans_p2c2b_stable_bank import (
    fit_bank,export_stable_evidence,bank_report)

def audit_input(evidence,k5a,manifest,old_rows,protocol_path,manifest_path):
    src=tuple(sorted(evidence['source_tasks']))
    outer=evidence['outer_target_excluded']
    if evidence.get('version')!='ktrans-empirical-competence-v1':
        raise ValueError('Expected original K5-A empirical evidence')
    if len(src)!=5 or outer in src or set(evidence.get('inner_partitions',()))!={'meta_SCUI','meta_UCUI'}:
        raise ValueError('Wrong source-only pseudo task protocol')
    if Path(evidence['k5a_protocol']).resolve()!=protocol_path.resolve():
        raise ValueError('K5-A provenance mismatch')
    if Path(evidence['manifest']).resolve()!=manifest_path.resolve():
        raise ValueError('Manifest provenance mismatch')
    if k5a['outer_task']!=outer or set(k5a['inner_source_tasks'])!=set(src):
        raise ValueError('K5-A source/outer task mismatch')
    if Path(k5a['outer_checkpoint']).resolve()!=Path(evidence['k4_checkpoint_for_provenance_only']).resolve():
        raise ValueError('Outer K4 provenance mismatch')
    if manifest.get('source_manifest_mode')!='outer-train-full-v1':
        raise ValueError('Use complete original OUTER TRAIN manifest')
    if manifest['heldout_task']!=outer or set(manifest['source_tasks'])!=set(src):
        raise ValueError('Outer heldout task in source manifest')
    if len(old_rows)!=int(evidence['n_expert_rows']):
        raise ValueError('Original empirical expert row count differs')
    folds=evidence.get('folds',{})
    if set(folds)!=set(src):raise ValueError('Incomplete inner LOTO fold metadata')
    cities=[]
    for task in src:
        fold=folds[task]
        if fold.get('checkpoint_choice')!='last.pt' or set(fold['inner_training_tasks'])!=set(src)-{task}:
            raise ValueError('Invalid original source-only fold')
        cities.append(frozenset(str(c) for c in fold['heldout_cities']))
    if not cities[0] or any(c!=cities[0] for c in cities):
        raise ValueError('LOTO folds do not share the SAME inner heldout city set')
    return src,outer,set(cities[0])

def load_source_records(manifest,nested_csv,feature_dir,image_root,remoteclip_model):
    """Exact original source manifest task/path pairs and cached features."""
    store=FeatureStore(feature_dir,encoder_id='RemoteCLIP-'+remoteclip_model)
    ds=CachedFeatureDataset(nested_csv,image_root,store,
        partition='eval',limit=None,task_dim=128)
    expected={(str(Path(r['image_path']).resolve()),r['task']):r
              for r in manifest['rows'] if r['partition']=='train'}
    if len(expected)!=len(manifest['rows']):
        raise ValueError('Duplicate/non-train original source manifest')
    if set(expected)!=set((str(Path(p).resolve()),str(row.indicator_column))
                         for p,row in zip(ds.paths,ds.df.itertuples(index=False))):
        raise ValueError('Complete source nested CSV != original outer train manifest')
    records=[]
    for i in range(len(ds)):
        item=ds[i]
        image_path=str(Path(item['image_path']).resolve())
        task=str(item['task_name'])
        city=str(item['city'])
        ref=expected[image_path,task]
        if city!=str(ref['city']):
            raise ValueError('Source manifest city mismatch')
        # Drop constant 4th context component because Ridge has bias.
        x=np.r_[item['image'].numpy().astype(float),
                item['context'].numpy().astype(float)[:3]]
        records.append({'task':task,'city':city,'image_path':image_path,
                        'x':x,'y':float(item['target'])})
    return records

def run(args):
    if args.alpha<=0 or args.competence_alpha<=0 or args.temperature<=0 or args.min_group<1:
        raise ValueError('Positive fixed hyperparameters required')
    evidence=json.loads((args.input/'protocol.json').read_text())
    k5a=json.loads(args.k5a_protocol.read_text())
    manifest=json.loads(args.manifest.read_text())
    old=[]
    with (args.input/'competence_rows.jsonl').open() as f:
        for line in f:
            if line.strip():old.append(json.loads(line))
    sources,outer,unseen=audit_input(
        evidence,k5a,manifest,old,args.k5a_protocol,args.manifest)
    checkpoint_path=Path(k5a['outer_checkpoint'])
    provenance=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    k4_meta=provenance['meta']
    if (k4_meta['heldout_task']!=outer or
        set(k4_meta['source_tasks'])!=set(sources)):
        raise ValueError('Original K4 checkpoint meta mismatch')
    clip_model=k4_meta['remoteclip_model']
    nested_csv=args.k5a_protocol.parent/'inner_source_only.csv'
    if not nested_csv.is_file():raise FileNotFoundError(nested_csv)
    texts,keys=text_descriptors(manifest,
        [args.reuse_cache,*args.extra_reuse_cache,args.knowledge_cache],
        args.model,args.max_pixels,sources)
    records=load_source_records(manifest,nested_csv,args.feature_dir,
                                args.image_root,clip_model)
    if len(records)!=len(manifest['rows']):
        raise ValueError('Wrong frozen source train record count')
    bank=fit_bank(records,sources,unseen,args.alpha)
    stable=export_stable_evidence(old,records,bank,sources,unseen)
    if len(stable)!=len(old):
        raise ValueError('Stable expert rows do not match old source evidence')
    # Group and verify paired predictions before writing any result.
    samples=aligned_samples(stable,outer,sources)
    source_partition_checks(samples,sources)
    args.output.mkdir(parents=True,exist_ok=True)
    evidence_dir=args.output/'stable_evidence'
    evidence_dir.mkdir(parents=True,exist_ok=True)
    with (evidence_dir/'competence_rows.jsonl').open('w') as f:
        for row in stable:f.write(json.dumps(row)+'\n')
    meta={'version':'ktrans-stable-independent-source-ridge-v1',
          'source_tasks':sources,'outer_target_excluded':outer,
          'original_evidence':str(args.input.resolve()),
          'source_manifest':str(args.manifest.resolve()),
          'k5a_protocol':str(args.k5a_protocol.resolve()),
          'nested_source_csv':str(nested_csv.resolve()),
          'remoteclip_encoder':'RemoteCLIP-'+clip_model,
          'source_training':'independent per-task frozen visual features + context, Ridge normalized on own source TRAIN city labels',
          'source_city_partition':'exact same historical nested unseen cities across all five pseudo folds',
          'unseen_cities':sorted(unseen),
          'training_city_labels':'ONLY each expert own source task, outside inner unseen city set',
          'model_selection':'fixed alpha and output temperature; no source UCUI/outer target tuning',
          'ridge_alpha':args.alpha,'competence_alpha':args.competence_alpha,
          'temperature':args.temperature,'seed':args.seed,
          'scale':'unweighted source-only mean/std average over four OTHER named source experts; same scale for all four predictions per pseudo-task',
          'fixed_expert_hashes':{name:bank[name]['hash'] for name in sources},
          'text_only_prompt_cache_keys':keys,
          'n_rows':len(stable),'n_tiles':len(samples),
          'dev_warning':'Exploratory only: K4 provenance best.pt outer Education meta-selected; source SCUI/UCUI have been viewed. Independent frozen Ridge experts are DIFFERENT architecture from earlier K4 learned shared neural experts; comparisons do not isolate the expert-identity effect.'
          }
    (args.output/'protocol.json').write_text(json.dumps(meta,indent=2))
    write_csv(args.output/'stable_expert_training.csv',bank_report(bank))
    print('[p2c2b] frozen_source_experts=',len(bank),
          'source_tiles=',len(records),'paired_eval_tiles=',len(samples),
          'expert_rows=',len(stable),flush=True)
    # Cross-source-task heldout ability estimator. NO new model
    # architecture: same P2-B1 fixed hyperparameters/feature definitions.
    cross=source_lopo(samples,sources,texts,args.competence_alpha,
                      args.temperature,args.seed)
    task_stats,macro=summarize_source_lopo(cross)
    write_csv(args.output/'stable_p2b1_source_task.csv',task_stats)
    write_csv(args.output/'stable_p2b1_macro.csv',macro)
    for v in macro:
        print(f'[p2c2b-cross] {v["model"]:<27} {v["inner_partition"]:<9} '
              f'top1={v["macro_top1_accuracy"]:.3f} '
              f'pairwise={v["macro_pairwise_accuracy"]:.3f} '
              f'mse_ratio_uniform={v["macro_mse_ratio_to_uniform"]:.3f}',flush=True)
    same=same_task_predictions(samples,sources,texts,args.competence_alpha,
                               args.temperature,args.seed)
    cross_ucui=[{**v,'route':'other_tasks_scui_to_ucui'} for v in cross
                if v['inner_partition']=='meta_UCUI']
    routes,route_macro=route_metrics(same+cross_ucui)
    write_csv(args.output/'stable_p2b2_task_ucui.csv',routes)
    write_csv(args.output/'stable_p2b2_macro.csv',route_macro)
    structure,pairs,ranking=expert_structure(samples,sources)
    write_csv(args.output/'stable_expert_structure.csv',structure)
    write_csv(args.output/'stable_residual_covariance.csv',pairs)
    write_csv(args.output/'stable_rank_stability.csv',ranking)
    # Re-test cross-task structure using the IDENTICAL frozen named
    # expert functions. The pseudo-own expert remains missing.
    names,tensor,masks,effects,thresholds,scales=standardize_effects(
        samples,sources,args.min_group)
    pair=task_pair_structure(names,tensor,masks,texts)
    svd=rank_structure(names,tensor,masks,'meta_SCUI')
    write_csv(args.output/'stable_cue_effects.csv',effects)
    write_csv(args.output/'stable_task_pair_ability_similarity.csv',pair)
    write_csv(args.output/'stable_masked_triplet_svd.csv',svd)
    for split in ('meta_SCUI','meta_UCUI'):
        selected=[v for v in pair if v['inner_partition']==split and
                  np.isfinite(v['competence_effect_cosine'])]
        cos=float(np.mean([v['competence_effect_cosine'] for v in selected])) if selected else float('nan')
        assoc=task_semantic_association([v for v in pair if v['inner_partition']==split])
        print(f'[p2c2b-structure] {split} task_pairs={len(selected)}/10 '
              f'mean_effect_cosine={cos:.3f} '
              f'text_vs_effect_pearson={assoc["pearson"]:.3f}',flush=True)
    print('[p2c2b] saved',args.output,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--k5a-protocol',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--knowledge-cache',type=Path,required=True)
    p.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--extra-reuse-cache',type=Path,action='append',default=[])
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--alpha',type=float,default=10.,help='Fixed source expert Ridge regularization')
    p.add_argument('--competence-alpha',type=float,default=10.)
    p.add_argument('--temperature',type=float,default=.5)
    p.add_argument('--min-group',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

if __name__=='__main__':main()
