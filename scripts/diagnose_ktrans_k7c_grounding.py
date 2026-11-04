"""Read-only paired grounding audit from K7-B originals and K7-C perturbations.

No target labels, model training or checkpoint selection. Changes indicate
pixel-input sensitivity, not correctness or causal transfer improvement.
"""
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_ktrans_k7_vlm_pilot import CUES,TASKS
from scripts.run_ktrans_k7b_controls import load_cached
from scripts.run_ktrans_k7c_grounding import transform_image,original_index,read_tile_manifest,CONDITIONS

def paired(a,b):
    result={}
    for field in ('visual_cues','evidence_relevance','evidence_confidence'):
        va=a[field];vb=b[field]
        diffs=[abs(va[c]-vb[c]) for c in CUES
               if va[c] is not None and vb[c] is not None]
        result[field+'_matched']=len(diffs)
        result[field+'_mean_abs_change']=float(np.mean(diffs)) if diffs else float('nan')
    return result

def summarize(original,perturbed,tiles,tasks):
    tile_map={t['tile_id']:t for t in tiles}
    rows=[]
    for rec in perturbed:
        if rec.get('condition') not in CONDITIONS or rec.get('tile_id') not in tile_map:
            continue
        key=(rec['tile_id'],rec['task'])
        if key not in original:continue
        base=original[key]
        row={'tile_id':rec['tile_id'],'city':rec['city'],'image_path':tile_map[rec['tile_id']]['image_path'],
             'task':rec['task'],'condition':rec['condition'],
             'baseline_valid':base.get('parsed') is not None,
             'intervention_valid':rec.get('parsed') is not None,
             'baseline_scene':base['parsed']['scene'] if base.get('parsed') else '',
             'intervention_scene':rec['parsed']['scene'] if rec.get('parsed') else '',
             'baseline_reasoning':base['parsed']['proxy_reasoning'] if base.get('parsed') else '',
             'intervention_reasoning':rec['parsed']['proxy_reasoning'] if rec.get('parsed') else '',
             'baseline_limitations':base['parsed']['limitations'] if base.get('parsed') else '',
             'intervention_limitations':rec['parsed']['limitations'] if rec.get('parsed') else '',
             # User must manually review original and altered images and assess grounding.
             'human_original_visual_fidelity':'',
             'human_intervention_visual_fidelity':'',
             'human_reasoning_supported':'',
             'human_notes':''}
        if base.get('parsed') and rec.get('parsed'):
            row.update(paired(base['parsed'],rec['parsed']))
        rows.append(row)
    return rows

def aggregate(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['task'],row['condition'])].append(row)
    summary=[]
    for (task,condition),group in sorted(groups.items()):
        valid=[r for r in group if r['baseline_valid'] and r['intervention_valid']]
        row={'task':task,'condition':condition,
             'paired_n':len(group),'paired_valid_n':len(valid)}
        for field in ('visual_cues','evidence_relevance','evidence_confidence'):
            col=field+'_mean_abs_change'
            numeric=[float(r[col]) for r in valid if col in r and np.isfinite(r[col])]
            row[col]=float(np.mean(numeric)) if numeric else float('nan')
        summary.append(row)
    return summary

def montage(tiles,folder,max_tiles=8):
    """Make local side-by-side source/blank/shuffle images for human inspection."""
    folder.mkdir(parents=True,exist_ok=True)
    for tile in tiles[:max_tiles]:
        with Image.open(tile['image_path']) as img:
            original=img.convert('RGB').resize((256,256))
        variants=[original]+[
            transform_image(original,condition,tile['tile_id']).resize((256,256))
            for condition in CONDITIONS]
        canvas=Image.new('RGB',(256*len(variants),286),'white')
        for i,(name,pic) in enumerate(zip(('original',)+CONDITIONS,variants)):
            canvas.paste(pic,(i*256,30))
            ImageDraw.Draw(canvas).text((i*256+5,7),name,fill='black')
        canvas.save(folder/(tile['tile_id']+'.png'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('runs/k7_pilot/vlm_evidence.jsonl'))
    p.add_argument('--original',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--interventions',type=Path,default=Path('runs/k7c_pilot/interventions.jsonl'))
    p.add_argument('--output',type=Path,default=Path('runs/k7c_pilot/diagnostics'))
    p.add_argument('--max-tiles',type=int,default=8)
    p.add_argument('--tasks',default=','.join(TASKS))
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--max-pixels',type=int,default=512*512)
    args=p.parse_args()
    tasks=tuple(x.strip() for x in args.tasks.split(',') if x.strip())
    tiles=read_tile_manifest(args.manifest,args.max_tiles)
    original=original_index(args.original,tiles,tasks,args.model,args.max_pixels)
    interventions=list(load_cached(args.interventions).values())
    rows=summarize(original,interventions,tiles,tasks)
    if not rows:p.error('No matched K7-C outputs')
    summary=aggregate(rows)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,table in [('paired_review.csv',rows),('task_summary.csv',summary)]:
        if table:
            keys=list(dict.fromkeys(k for row in table for k in row))
            with (args.output/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=keys)
                w.writeheader();w.writerows(table)
    montage(tiles,args.output/'image_review')
    print(f'[k7c] paired_records={len(rows)} image_tiles={len(tiles)}',flush=True)
    for row in summary:
        print(f"[k7c] {row['task']:<32} {row['condition']:<14} "
              f"valid={row['paired_valid_n']}/{row['paired_n']} "
              f"cue_delta={row['visual_cues_mean_abs_change']:.3f} "
              f"relevance_delta={row['evidence_relevance_mean_abs_change']:.3f} "
              f"confidence_delta={row['evidence_confidence_mean_abs_change']:.3f}",flush=True)
    print('[k7c] Image-only perturbations test VISUAL SENSITIVITY, not label validity. '
          'Human inspection of paired_review.csv and image_review PNG files is essential.',flush=True)
    print('[k7c] saved to',args.output,flush=True)

if __name__=='__main__':main()
