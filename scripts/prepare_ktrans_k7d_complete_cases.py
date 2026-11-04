"""Generate a label-free complete-case K7-D manifest from valid VLM evidence.

Keep a row only if its image_only, image_task, and text_only
evidence are all valid for the exact existing prompt/model/cache key.
The same filtered train and evaluation rows are shared by all four
ablations. Never impute, fabricate, or use target values to filter.
"""
import argparse,collections,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.cache_ktrans_k7d import unique_jobs,merge_records
from scripts.prepare_ktrans_k7d import PARTITIONS

def complete_cases(manifest,records,model,max_pixels):
    all_jobs=unique_jobs(manifest,model,max_pixels)
    indexed={(j['tile_id'],j['task'],j['condition']):j for j in all_jobs}
    valid={}
    for key,job in indexed.items():
        row=records.get(job['key'])
        valid[key]=bool(row and row.get('parsed') is not None and
              row.get('condition')==job['condition'] and
              row.get('task')==job['task'] and
              row.get('prompt')==job['prompt'] and
              row.get('model')==job['model'] and
              row.get('max_pixels')==job['max_pixels'])
    kept=[];removed=[];reasons=collections.Counter()
    for row in manifest['rows']:
        tile,task=row['tile_id'],row['task']
        required=((tile,'image_only','image_only'),
                  (tile,task,'image_task'),
                  (None,task,'text_only'))
        missing=[x[2] for x in required if not valid[x]]
        if missing:
            removed.append({**row,'missing_conditions':missing})
            for reason in missing:reasons[reason]+=1
        else:kept.append(row)
    counts={p:sum(row['partition']==p for row in kept) for p in PARTITIONS}
    original_counts={p:sum(row['partition']==p for row in manifest['rows']) for p in PARTITIONS}
    if any(counts[p]==0 for p in PARTITIONS):
        raise ValueError('Complete-case filtering would empty a partition; refuse training')
    train_tasks={r['task'] for r in kept if r['partition']=='train'}
    if train_tasks!=set(manifest['source_tasks']):
        raise ValueError('Complete-case filtering would eliminate a source train task')
    if counts['train']<4*len(train_tasks):
        raise ValueError('Not enough source training samples for router')
    new={**manifest,'rows':kept,'partition_counts':counts,
         'complete_case_protocol':{
           'version':'k7d-complete-case-v1',
           'selection':'Only rows with all three valid VLM evidence conditions; label-free',
           'original_counts':original_counts,
           'removed_counts':{p:original_counts[p]-counts[p] for p in PARTITIONS},
           'removed_rows':len(removed),
           'missing_conditions_row_counts':dict(reasons),
           'valid_cache_keys':len([j for j in all_jobs if valid[(j['tile_id'],j['task'],j['condition'])]]),
           'total_required_cache_keys':len(all_jobs),
           'warning':'DEV ONLY; missing VLM outputs may not be MCAR. Compare all variants on this same filtered subset; do not compare to full-data K5-A metrics.'
         }}
    # There must be no needed invalid key for any surviving row.
    for row in kept:
        tile,task=row['tile_id'],row['task']
        for key in ((tile,'image_only','image_only'),(tile,task,'image_task'),
                    (None,task,'text_only')):
            if not valid[key]:raise AssertionError('Filtered row still needs missing knowledge')
    return new,removed

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--max-pixels',type=int,default=262144)
    args=p.parse_args()
    if args.output.resolve()==args.manifest.resolve():
        p.error('Do not overwrite the original manifest')
    manifest=json.loads(args.manifest.read_text())
    if manifest.get('version')!='k7d-manifest-v1':p.error('Unsupported K7-D manifest')
    records=merge_records([args.reuse_cache,args.cache])
    filtered,removed=complete_cases(manifest,records,args.model,args.max_pixels)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(filtered,indent=2))
    audit=args.output.with_suffix('.excluded.json')
    audit.write_text(json.dumps({
        'version':'k7d-complete-case-v1',
        'original_manifest':str(args.manifest.resolve()),
        'cache':str(args.cache.resolve()),
        'removed_rows':removed
    },indent=2))
    print('[k7d-complete-case] original=',len(manifest['rows']),
          'kept=',len(filtered['rows']),'excluded=',len(removed),flush=True)
    print('[k7d-complete-case] original_partitions=',
          filtered['complete_case_protocol']['original_counts'],flush=True)
    print('[k7d-complete-case] filtered_partitions=',
          filtered['partition_counts'],flush=True)
    print('[k7d-complete-case] missing_conditions=',
          filtered['complete_case_protocol']['missing_conditions_row_counts'],flush=True)
    print('[k7d-complete-case] saved manifest=',args.output,'audit=',audit,flush=True)

if __name__=='__main__':main()
