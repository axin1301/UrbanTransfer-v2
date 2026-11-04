"""Export source-only, cross-task empirical expert evidence from existing K5-A INNER folds.

No new training or VLM inference. We require inner last.pt and only
predict INNER pseudo-heldout task on meta_SCUI and meta_UCUI, using
fixed source-only semantic output moments. Outer K4 checkpoint is
loaded for identity/provenance ONLY, not to make competence predictions.
"""
import argparse,collections,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_k4_profiles import load_model
from scripts.run_ktrans_k5a import extract_outputs
from scripts.run_ktrans_k7d import validate_manifest
from urban_transfer.ktrans_source_coverage import validate_full_source_manifest
from scripts.cache_ktrans_k7d import merge_records
from urban_transfer.ktrans_empirical_competence import (
    identity,manifest_source_index,validate_inner_fold,
    validated_vlm_jobs,assemble_sample_rows,audit_identity)

INNER_SPLITS=('meta_SCUI','meta_UCUI')

def build_fold(pseudo,fold_path,outer_train_keys,manifest_rows,outer_sources,
               outer_target,nested_csv,vlm,args):
    if fold_path.name!='last.pt':raise ValueError('Use inner last.pt, NEVER best.pt')
    meta,ds,splits,bank,device=load_model(fold_path,args.feature_dir,
                                          args.text_cache_dir,args.image_root)
    validate_inner_fold(pseudo,meta,ds,splits,bank,outer_train_keys,
                        outer_sources,outer_target,nested_csv)
    all_rows=[];reports=[]
    for split in INNER_SPLITS:
        indices=[int(i) for i in splits[split]]
        # Validate split membership before intersecting with VLM cache.
        match=[];selected=[]
        for ix in indices:
            key=identity(ds.base.paths[ix],ds.df.iloc[ix].indicator_column)
            if key in manifest_rows:
                match.append(manifest_rows[key])
                selected.append(ix)
        counts={'pseudo_task':pseudo,'inner_partition':split,
             'total_pseudo_rows':len(indices),'matched_outer_manifest_rows':len(selected),
             'not_in_outer_manifest':len(indices)-len(selected)}
        if not selected:
            counts.update(with_vlm=0,missing_image_task_vlm=0,
                          competence_rows=0,experts=len(bank.source_tasks))
            reports.append(counts)
            continue
        if any(row['task']!=pseudo for row in match):
            raise AssertionError('Pseudo-task match contaminated')
        z,y=extract_outputs(bank,ds,selected,device,args.batch_size)
        # extract_outputs is [n_experts,n_rows], requiring transpose.
        rows,coverage=assemble_sample_rows(pseudo,split,match,z.T,y,bank,vlm)
        counts.update(coverage)
        counts.setdefault('with_vlm',0)
        counts.setdefault('missing_image_task_vlm',0)
        counts.update(competence_rows=len(rows),experts=len(bank.source_tasks))
        if len(rows)!=counts['with_vlm']*len(bank.source_tasks):
            raise AssertionError('Incomplete per-expert prediction evidence')
        all_rows.extend(rows);reports.append(counts)
        print(f'[competence] pseudo={pseudo} split={split} '
              f'inner={len(indices)} matched={len(selected)} '
              f'vlm={counts["with_vlm"]} expert_rows={len(rows)}',flush=True)
    return all_rows,reports,meta

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','k5a-protocol','manifest','knowledge-cache','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    parser.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--max-pixels',type=int,default=262144)
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--feature-dir',default='runs/m3_features')
    parser.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    parser.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    a=parser.parse_args()
    if a.batch_size<1:parser.error('Batch size must be positive')
    info=json.loads(a.k5a_protocol.read_text())
    manifest=json.loads(a.manifest.read_text())
    outer,ds,splits,bank,_=load_model(a.checkpoint,a.feature_dir,
                                      a.text_cache_dir,a.image_root)
    if manifest.get('source_manifest_mode')=='outer-train-full-v1':
        validate_full_source_manifest(manifest,ds,splits,outer,a.checkpoint)
    else:
        validate_manifest(manifest,ds,splits,outer,a.checkpoint)
    sources=list(bank.source_tasks)
    target=outer['heldout_task']
    if Path(info['outer_checkpoint']).resolve()!=a.checkpoint.resolve() or (
        info['outer_task']!=target or list(info['inner_source_tasks'])!=sources):
        raise ValueError('K5-A outer checkpoint/task/source order mismatch')
    if a.checkpoint.name=='best.pt':
        print('[competence] WARNING: K4 best.pt was DEV-selected using outer meta labels; this export is DEV only',flush=True)
    nested_csv=a.k5a_protocol.parent/'inner_source_only.csv'
    if not nested_csv.is_file():raise FileNotFoundError(nested_csv)
    outer_train_keys=set()
    outer_cities=set()
    for ix in splits['train']:
        key=identity(ds.base.paths[int(ix)],ds.df.iloc[int(ix)].indicator_column)
        if key[1]==target:raise AssertionError('Outer heldout in train')
        if key in outer_train_keys:raise ValueError('Duplicated outer train image/task')
        outer_train_keys.add(key)
        outer_cities.add(str(ds.df.iloc[int(ix)].city))
    outer_manifest=manifest_source_index(manifest)
    if not set(outer_manifest).issubset(outer_train_keys):
        raise ValueError('VLM outer train manifest contains rows outside checkpoint OUTER TRAIN')
    vlm=validated_vlm_jobs(manifest,
                          merge_records([a.reuse_cache,a.knowledge_cache]),
                          a.model,a.max_pixels)
    all_rows=[];reports=[];folds={}
    for pseudo in sources:
        path=a.k5a_protocol.parent/'inner_models'/pseudo/'last.pt'
        if not path.is_file():raise FileNotFoundError('Missing K5-A inner last checkpoint: '+str(path))
        rows,stats,meta=build_fold(pseudo,path,outer_train_keys,outer_manifest,
             sources,target,nested_csv,vlm,a)
        if any(row['city'] not in outer_cities for row in rows):
            raise ValueError('Inner evidence includes outer heldout city')
        all_rows.extend(rows);reports.extend(stats)
        folds[pseudo]={'checkpoint':str(path.resolve()),
                       'heldout_cities':meta['heldout_cities'],
                       'inner_training_tasks':meta['source_tasks'],
                       'checkpoint_choice':'last.pt'}
    audit_identity(all_rows,target)
    if not all_rows:raise ValueError('No cached source-only pseudo-task evidence matched inner folds')
    # Unique records contain source pseudo-task labels only; no heldout
    # target label is read by the sample extraction path.
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'competence_rows.jsonl').open('w') as f:
        for row in all_rows:f.write(json.dumps(row)+'\n')
    (a.output/'coverage.json').write_text(json.dumps(reports,indent=2))
    (a.output/'protocol.json').write_text(json.dumps({
       'version':'ktrans-empirical-competence-v1',
       'source_tasks':sources,'outer_target_excluded':target,
       'k4_checkpoint_for_provenance_only':str(a.checkpoint.resolve()),
       'k5a_protocol':str(a.k5a_protocol.resolve()),
       'manifest':str(a.manifest.resolve()),
       'source_manifest_mode':manifest.get('source_manifest_mode','sampled-k7d'),
       'folds':folds,'n_expert_rows':len(all_rows),
       'inner_partitions':list(INNER_SPLITS),
       'spatial_label_mode':'log1p source pseudo-task labels',
       'per_expert_scale':'inner pseudo-target semantic route @ inner source-only moments (identical for all experts per pseudo fold)',
       'comparison_protocol':'meta_SCUI used in K5-A ridge scoring; meta_UCUI inner unseen-city heldout source-task rows',
       'missing_vlm_policy':'only matching valid prompt-key image_task used; report all missing separately',
       'independent_validation_certified':False,
       'warning':'DEV ONLY, K4 best.pt selected on outer meta. Inner SCUI labels already informed K5-A ridge. No outer target labels used in empirical competence. Inner UCUI is pseudo-task unseen city, not independent final test.'
    },indent=2))
    print('[competence] exported expert_rows=',len(all_rows),
          'matched_tiles=',len(set((r['pseudo_task'],r['image_path'],r['inner_partition']) for r in all_rows)),
          'output=',a.output,flush=True)

if __name__=='__main__':main()
