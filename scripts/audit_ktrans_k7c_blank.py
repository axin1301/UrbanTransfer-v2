"""K7-C quality gate: distinguish missing visual evidence from invalid JSON.

Read-only analysis of cached K7-C blank and shuffled interventions.
Do not interpret all-null as zero score or automatically certify hallucination.
Human reviewers must inspect scene text against image triptychs separately.
"""
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_ktrans_k7_vlm_pilot import CUES,TASKS
from scripts.run_ktrans_k7b_controls import load_cached

FIELDS=('visual_cues','evidence_relevance','evidence_confidence')

def summarize_blank(records,tasks=TASKS):
    grouped=defaultdict(list)
    for record in records:
        if record.get('condition')=='blank' and record.get('task') in tasks:
            grouped[record['task']].append(record)
    rows=[]
    review=[]
    for task in tasks:
        batch=grouped[task]
        valid=[r for r in batch if r.get('parsed') is not None]
        row={'task':task,'total_records':len(batch),'valid_json':len(valid)}
        for field in FIELDS:
            values=[r['parsed'][field].get(cue) for r in valid for cue in CUES]
            nonnull=[value for value in values if value is not None]
            row[field+'_nonnull']=len(nonnull)
            row[field+'_total']=len(values)
            row[field+'_nonnull_fraction']=len(nonnull)/len(values) if values else None
            row[field+'_positive']=sum(value>0 for value in nonnull)
        row['blank_visual_fidelity_auto']='NOT_ASSESSED'
        row['manual_review_required']=True
        rows.append(row)
        for r in batch:
            parsed=r.get('parsed') or {}
            review.append({
                'tile_id':r.get('tile_id'),'city':r.get('city'),'task':task,
                'condition':'blank','valid_json':r.get('parsed') is not None,
                'scene':parsed.get('scene',''),
                'proxy_reasoning':parsed.get('proxy_reasoning',''),
                'limitations':parsed.get('limitations',''),
                'visual_cues':json.dumps(parsed.get('visual_cues'),ensure_ascii=False),
                'evidence_relevance':json.dumps(parsed.get('evidence_relevance'),ensure_ascii=False),
                'evidence_confidence':json.dumps(parsed.get('evidence_confidence'),ensure_ascii=False),
                'human_hallucinated_visible_structure':'',
                'human_confidence_justified':'',
                'human_explanation_consistent':'',
                'human_notes':'',
            })
    return rows,review

def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('runs/k7c_pilot/interventions.jsonl'))
    p.add_argument('--output',type=Path,default=Path('runs/k7c_pilot/quality_gate'))
    args=p.parse_args()
    if not args.input.exists():p.error('Missing K7-C intervention cache')
    records=list(load_cached(args.input).values())
    summary,review=summarize_blank(records)
    if not review:p.error('No blank interventions in cache')
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'blank_summary.csv',summary)
    write_csv(args.output/'blank_human_review.csv',review)
    flags=[]
    for row in summary:
        if row['valid_json']<row['total_records']:
            flags.append(row['task']+': incomplete valid JSON')
        if row['visual_cues_positive'] or row['evidence_confidence_positive']:
            flags.append(row['task']+': positive visual claims despite blank image; inspect')
        print(f"[k7c-quality] {row['task']:<32} valid={row['valid_json']}/{row['total_records']} "
            f"cue_nonnull={row['visual_cues_nonnull']}/{row['visual_cues_total']} "
            f"relevance_nonnull={row['evidence_relevance_nonnull']}/{row['evidence_relevance_total']} "
            f"confidence_nonnull={row['evidence_confidence_nonnull']}/{row['evidence_confidence_total']}",
            flush=True)
    report={
        'blank_records':len(review),'status':'PENDING_HUMAN_GROUNDING_REVIEW',
        'positive_visual_claims_flags':flags,
        'definitions':{
          'nonnull':'a numerical 0-3 rating; null excluded and never imputed as 0',
          'all_null':'unassessable scores; cannot compute original-vs-blank paired difference',
          'positive':'non-null rating greater than 0; requires manual checking on blank image'},
        'caveat':'No source or heldout indicator labels are used. No automatic pass/fail of VLM semantic fidelity.',
        'next_steps':'Review paired_review.csv and image_review/*.png from K7-C diagnostics; document any unsupported visual claims.',
    }
    (args.output/'readiness.json').write_text(json.dumps(report,indent=2))
    for flag in flags:print('[k7c-quality][REVIEW]',flag,flush=True)
    print('[k7c-quality] human visual review still required; saved',args.output,flush=True)

if __name__=='__main__':main()
