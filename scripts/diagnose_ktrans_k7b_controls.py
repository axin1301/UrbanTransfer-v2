"""Read-only K7-B factorial analysis; never use urban indicator labels.

Key separation: task-only relevance prior vs. the additional image-dependent
relevance shift. Summarizes all task/condition combinations, including
incomplete/invalid JSON, and does NOT claim predictive improvement.
"""
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_ktrans_k7b_controls import CUES,TASKS,load_cached

def mean_or_nan(values):
    return float(np.mean(values)) if values else float('nan')

def compute(records,tasks=TASKS):
    """Return per-task and per-tile summaries from paired valid outputs."""
    text={}
    pairs=defaultdict(dict)
    image_only={}
    for row in records:
        if row.get('condition')=='text_only':
            text[row['task']]=row
        elif row.get('condition')=='image_task':
            pairs[row['tile_id']][row['task']]=row
        elif row.get('condition')=='image_only':
            image_only[row['tile_id']]=row
    per_task=[]
    for task in tasks:
        prior=text.get(task)
        prior_val=prior.get('parsed') if prior else None
        total=0;valid=0;matched=0;image_shifts=[];visual_variation=[]
        values_by_cue=defaultdict(list)
        for tile,rows in pairs.items():
            if task not in rows:continue
            total+=1
            obj=rows[task].get('parsed')
            if obj is None:continue
            valid+=1
            for cue in CUES:
                val=obj['evidence_relevance'][cue]
                if val is not None:values_by_cue[cue].append(val)
            if prior_val is not None:
                for cue in CUES:
                    a=obj['evidence_relevance'][cue]
                    b=prior_val['evidence_relevance'][cue]
                    if a is not None and b is not None:
                        image_shifts.append(abs(a-b));matched+=1
        for cue in CUES:
            vals=values_by_cue[cue]
            if len(vals)>=2:visual_variation.append(float(np.std(vals)))
        per_task.append({'task':task,'expected_tile_conditions':len(pairs),
           'present_tile_conditions':total,'valid_image_task':valid,
           'valid_text_only':prior_val is not None,'matched_cue_pairs':matched,
           'mean_abs_image_vs_text_relevance':mean_or_nan(image_shifts),
           'mean_within_task_across_image_std':mean_or_nan(visual_variation),
           'valid_relevance_cue_fraction':sum(len(v) for v in values_by_cue.values())/
                (valid*len(CUES)) if valid else 0.})
    per_tile=[]
    for tile,rows in sorted(pairs.items()):
        all_valid=[rows[t]['parsed'] for t in tasks
                   if t in rows and rows[t].get('parsed') is not None]
        distinct_tasks=sum(t in rows for t in tasks)
        deltas=[];task_variation=[];visual_cue_var=[]
        for cue in CUES:
            vals=[d['evidence_relevance'][cue] for d in all_valid
                  if d['evidence_relevance'][cue] is not None]
            if len(vals)>=2:task_variation.append(float(np.std(vals)))
            visuals=[d['visual_cues'][cue] for d in all_valid
                     if d['visual_cues'][cue] is not None]
            if len(visuals)>=2:visual_cue_var.append(float(np.std(visuals)))
        for task in tasks:
            rec=rows.get(task)
            baseline=text.get(task)
            if not rec or not baseline or not rec.get('parsed') or not baseline.get('parsed'):
                continue
            a=rec['parsed']['evidence_relevance']
            b=baseline['parsed']['evidence_relevance']
            for cue in CUES:
                if a[cue] is not None and b[cue] is not None:
                    deltas.append(abs(a[cue]-b[cue]))
        per_tile.append({'tile_id':tile,
            'city':next(iter(rows.values())).get('city'),
            'present_tasks':distinct_tasks,'valid_image_task':len(all_valid),
            'image_only_valid':bool(image_only.get(tile,{}).get('parsed')),
            'task_relevance_std':mean_or_nan(task_variation),
            'visual_cues_across_task_std':mean_or_nan(visual_cue_var),
            'mean_abs_image_vs_text_relevance':mean_or_nan(deltas)})
    return per_task,per_tile

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--output',type=Path,default=Path('runs/k7b_pilot/diagnostics'))
    p.add_argument('--tasks',default=','.join(TASKS))
    args=p.parse_args()
    if not args.input.is_file():p.error('Missing K7-B evidence cache')
    tasks=tuple(x.strip() for x in args.tasks.split(',') if x.strip())
    records=list(load_cached(args.input).values())
    task_rows,tile_rows=compute(records,tasks)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('task_summary.csv',task_rows),('tile_summary.csv',tile_rows)]:
        if not rows:continue
        with (args.output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    print(f'[k7b] cached_records={len(records)} tiles={len(tile_rows)} tasks={len(task_rows)}')
    for row in task_rows:
        print(f"{row['task']:<32} image={row['valid_image_task']}/{row['expected_tile_conditions']} "
              f"text_only={row['valid_text_only']} "
              f"cue_coverage={row['valid_relevance_cue_fraction']:.3f} "
              f"image_vs_text_abs={row['mean_abs_image_vs_text_relevance']:.3f} "
              f"across_image_std={row['mean_within_task_across_image_std']:.3f}")
    print('[k7b] task differences can be prompt-driven; the paired text-only control '
          'is necessary but not sufficient to show image grounding.',flush=True)
    print('[k7b] no target labels or predictive performance used. Saved to',args.output,flush=True)

if __name__=='__main__':main()
