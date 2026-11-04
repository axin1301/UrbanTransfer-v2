"""Repair an existing K7-D cache by strict, auditable Image-Only v2 projection.

Append success records with the SAME versioned key; do not mutate
raw outputs, prior successful records, or other experimental conditions.
Only an explicitly whitelisted set of extra VLM fields can be discarded.
Incomplete outputs remain pending for GPU regeneration.
"""
import argparse,csv,json,os,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.cache_ktrans_k7d import unique_jobs,merge_records
from urban_transfer.ktrans_k7d_recovery import project_image_only_v2
from urban_transfer.ktrans_k7d_image_only import IMAGE_ONLY_VERSION

def recover(manifest,cache,reuse,model,max_pixels):
    jobs=[j for j in unique_jobs(manifest,model,max_pixels)
          if j['condition']=='image_only']
    records=merge_records([reuse,cache])
    add=[];audit=[];summary=Counter()
    for job in jobs:
        row=records.get(job['key'])
        if row is None:
            summary['missing']+=1
            audit.append({'tile_id':job['tile_id'],'status':'missing',
                          'discarded_fields':'','raw_chars':0})
            continue
        if row.get('parsed') is not None:
            summary['already_valid']+=1
            continue
        if (row.get('version')!=IMAGE_ONLY_VERSION or
            row.get('prompt')!=job['prompt'] or
            row.get('model')!=job['model'] or
            row.get('max_pixels')!=job['max_pixels'] or
            row.get('image_path')!=job['image_path']):
            raise ValueError('Cache provenance mismatch for key '+job['key'])
        raw=row.get('raw','')
        parsed,extras,reason=project_image_only_v2(raw)
        # A top-level null confidence denotes all six missing confidences.
        # Record the equivalence explicitly so the schema normalization
        # cannot be mistaken for hallucinated or zero-valued evidence.
        whole_null_confidence=False
        try:
            whole_null_confidence=json.loads(raw).get('evidence_confidence','MISSING') is None
        except (ValueError,AttributeError):
            pass
        if parsed is not None and whole_null_confidence:
            reason='expanded_null_confidence' if reason=='strict' else reason+'+expanded_null_confidence'
        audit.append({'tile_id':job['tile_id'],'status':reason,
                      'discarded_fields':';'.join(extras),
                      'raw_chars':len(raw)})
        summary[reason]+=1
        if parsed is not None:
            record={**row,'parsed':parsed,'parse_status':reason,
                    'discarded_extra_fields':list(extras),
                    'recovery_policy':'k7d-image-only-projection-v2',
                    'whole_null_confidence_expanded':whole_null_confidence}
            add.append(record)
    return add,audit,summary,len(jobs)

def write_repair(cache,add):
    if not add:return
    cache.parent.mkdir(parents=True,exist_ok=True)
    with cache.open('a') as f:
        for row in add:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--audit',type=Path)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    if manifest.get('version')!='k7d-manifest-v1':
        p.error('Expected K7-D v1 sample manifest')
    add,audit,summary,total=recover(manifest,args.cache,args.reuse_cache,
                                     args.model,args.max_pixels)
    if not args.dry_run:
        write_repair(args.cache,add)
    path=args.audit or args.cache.parent/'image_only_recovery_audit.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    if audit:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(audit[0]))
            w.writeheader();w.writerows(audit)
    print('[k7d-repair] image_only_total=',total,
          'projected_valid=',len(add),
          'remaining_unresolved=',sum(summary[k] for k in (
              'missing','invalid_json','invalid_object','unexpected_schema',
              'invalid_visual_scores')),flush=True)
    for status,count in sorted(summary.items()):
        print(f'[k7d-repair] {status}={count}',flush=True)
    if args.dry_run:
        print('[k7d-repair] dry-run; no cache entries appended',flush=True)
    print('[k7d-repair] audit=',path,flush=True)

if __name__=='__main__':main()
