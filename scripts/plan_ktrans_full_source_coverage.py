"""Plan full outer-source-TRAIN VLM coverage and cache reuse; NO inference.

Writes a source-only manifest accepted by existing cache_ktrans_k7d.py
and a detailed report. NEVER reads heldout task labels for selection.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_k4_profiles import load_model
from scripts.cache_ktrans_k7d import merge_records
from urban_transfer.ktrans_source_coverage import (
    full_source_manifest,coverage_plan)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--existing-manifest',type=Path,
                   default=Path('runs/k7d_education/manifest_complete_cases.json'))
    p.add_argument('--knowledge-cache',required=True,type=Path)
    p.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--additional-cache',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=p.parse_args()
    meta,ds,splits,_,_=load_model(args.checkpoint,args.feature_dir,
                                  args.text_cache_dir,args.image_root)
    full=full_source_manifest(ds,splits,meta,args.checkpoint)
    old=json.loads(args.existing_manifest.read_text())
    if Path(old['k4_checkpoint']).resolve()!=args.checkpoint.resolve() or (
        old['heldout_task']!=meta['heldout_task']):
        raise ValueError('Existing manifest belongs to another checkpoint/outer task')
    old_keys={(r['image_path'],r['task']) for r in old['rows'] if r['partition']=='train'}
    paths=[args.reuse_cache,args.knowledge_cache]
    if args.additional_cache:paths.append(args.additional_cache)
    stats,pending=coverage_plan(full,merge_records(paths),args.model,
                                args.max_pixels,old_keys)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'source_train_manifest.json').write_text(json.dumps(full,indent=2))
    (args.output/'coverage_plan.json').write_text(json.dumps({
       'version':'ktrans-source-full-coverage-plan-v1',
       'k4_checkpoint':str(args.checkpoint.resolve()),
       'model':args.model,'max_pixels':args.max_pixels,
       'existing_manifest':str(args.existing_manifest.resolve()),
       'input_caches':[str(Path(x).resolve()) for x in paths],
       **stats},indent=2))
    with (args.output/'source_task_coverage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(stats['source_task_coverage'][0]))
        w.writeheader();w.writerows(stats['source_task_coverage'])
    # Pending jobs list is for budgeting. Exact full manifest is always
    # used as the cache-run input so key/prompt generation stays canonical.
    with (args.output/'pending_jobs.jsonl').open('w') as f:
        for j in pending:f.write(json.dumps(j,ensure_ascii=False)+'\n')
    print('[source-coverage] outer_train_rows=',stats['rows'],
          'unique_tiles=',stats['unique_tiles'],
          'unique_image_task_pairs=',stats['unique_image_task_pairs'],flush=True)
    print('[source-coverage] total_jobs=',stats['jobs_total'],
          'cached_valid=',stats['jobs_cached_valid'],
          'pending=',stats['jobs_pending'],flush=True)
    for row in stats['source_task_coverage']:
        print('[source-coverage]',row['task'],
              'full=',row['total_train_rows'],
              'old_manifest=',row['in_previous_manifest'],
              'cached_image_task=',row['cached_image_task'],
              'missing=',row['pending_image_task'],flush=True)
    print('[source-coverage] no VLM inference performed; outputs=',args.output,flush=True)
if __name__=='__main__':main()
