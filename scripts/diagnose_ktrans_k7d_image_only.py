"""Classify K7-D v2 inference failures from saved raw output; no GPU required."""
import argparse,csv,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.cache_ktrans_k7d import unique_jobs,merge_records
from urban_transfer.ktrans_k7d_recovery import project_image_only_v2

def inspect(manifest,records,model,max_pixels):
    jobs=[j for j in unique_jobs(manifest,model,max_pixels)
          if j['condition']=='image_only']
    rows=[]
    for job in jobs:
        record=records.get(job['key'])
        if record is None:
            reason='never_generated';raw='';extras=()
        elif record.get('parsed') is not None:
            reason='valid';raw=record.get('raw','');extras=()
        else:
            raw=record.get('raw','')
            result,extras,reason=project_image_only_v2(raw)
            if result is not None:reason='projectable'
        rows.append({
            'tile_id':job['tile_id'],'task':job['task'],'reason':reason,
            'raw_chars':len(raw),'raw_tail':raw[-160:].replace('\n','\\n'),
            'discarded_extra_fields':';'.join(extras),
        })
    return rows,Counter(row['reason'] for row in rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--output',type=Path,default=Path('runs/k7d_education/image_only_failure_report.csv'))
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    records=merge_records([args.reuse_cache,args.cache])
    rows,counts=inspect(manifest,records,args.model,args.max_pixels)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    for reason,count in sorted(counts.items()):
        print(f'[k7d-diagnose] {reason}={count}',flush=True)
    bad=[r for r in rows if r['reason'] not in ('valid','projectable')]
    for row in bad[:5]:
        print('[k7d-diagnose] example',row,flush=True)
    print('[k7d-diagnose] saved',args.output,flush=True)

if __name__=='__main__':main()
