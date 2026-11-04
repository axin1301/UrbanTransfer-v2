"""Summarize K7 8B VLM pilot: parse yield and within-tile task dependence.

Only cached prompts and model outputs are accessed; target labels are never read.
Variation measures are descriptive, NOT evidence of prediction improvement.
"""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scripts.run_ktrans_k7_vlm_pilot import CUES

def analyze(records):
    by_tile=defaultdict(dict)
    for rec in records:
        if rec['task'] in by_tile[rec['tile_id']]:
            # Latest attempt takes precedence when rerunning failed JSON.
            pass
        by_tile[rec['tile_id']][rec['task']]=rec
    output=[]
    for tile,items in sorted(by_tile.items()):
        valid={task:rec['parsed'] for task,rec in items.items() if rec.get('parsed')}
        tasks=[t for t in valid if t!='image_only']
        relevance=[]
        scene_variations=[]
        for cue in CUES:
            vals=[valid[t]['task_relevance'][cue] for t in tasks
                  if valid[t]['task_relevance'][cue] is not None]
            if len(vals)>=2:relevance.append(float(np.std(vals)))
            visual_vals=[valid[t]['visual_cues'][cue] for t in tasks
                  if valid[t]['visual_cues'][cue] is not None]
            if len(visual_vals)>=2:scene_variations.append(float(np.std(visual_vals)))
        output.append({
            'tile_id':tile,
            'city':next(iter(items.values()))['city'],
            'conditions_present':len(items),
            'valid_json':len(valid),
            'n_valid_task_conditions':len(tasks),
            'mean_task_relevance_std':float(np.mean(relevance)) if relevance else None,
            'mean_task_visual_cue_std':float(np.mean(scene_variations)) if scene_variations else None,
            'distinct_scene_descriptions':len({valid[t]['scene'].strip().lower() for t in tasks}),
        })
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('runs/k7_pilot/vlm_evidence.jsonl'))
    p.add_argument('--output',type=Path,default=Path('runs/k7_pilot/summary.csv'))
    args=p.parse_args()
    if not args.input.is_file():p.error('Missing VLM cache')
    with args.input.open() as f:
        records=[json.loads(line) for line in f if line.strip()]
    result=analyze(records)
    if not result:p.error('No cached VLM results')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]))
        w.writeheader();w.writerows(result)
    print('tiles=',len(result),'records=',len(records),
          'valid_json=',sum(r['valid_json'] for r in result),flush=True)
    for row in result:
        print(f"{row['city']:<22} valid={row['valid_json']}/{row['conditions_present']} "
              f"task_relevance_std={row['mean_task_relevance_std']} "
              f"visual_cue_std={row['mean_task_visual_cue_std']} "
              f"distinct_task_descriptions={row['distinct_scene_descriptions']}",flush=True)
    print('Saved',args.output,flush=True)

if __name__=='__main__':main()
