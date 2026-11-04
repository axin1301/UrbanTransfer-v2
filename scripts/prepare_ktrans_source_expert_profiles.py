"""Prepare auditable source-only six-cue expert-proxy profiles.

No targets from the OUTER heldout task are needed or read. A profile
encodes correlations of each source expert on INNER pseudo-heldout
source tasks combined with those pseudo tasks' text-only conceptual
urban cue relevance. This proxy is NOT proven visual competence.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.cache_ktrans_k7d import unique_jobs,merge_records
from urban_transfer.ktrans_knowledge_expert_alignment import (
    build_expert_cue_proxy,read_transfer_scores)

def prepare(manifest,protocol,score_rows,records,model,max_pixels):
    sources=list(manifest['source_tasks'])
    outer=manifest['heldout_task']
    if protocol['outer_task']!=outer or list(protocol['inner_source_tasks'])!=sources:
        raise ValueError('K5-A/manifest target and source-task ordering mismatch')
    if Path(protocol['outer_checkpoint']).resolve()!=Path(manifest['k4_checkpoint']).resolve():
        raise ValueError('K5-A/manifest K4 mismatch')
    if outer in sources or any(row['pseudo_target']==outer or row['source_expert']==outer for row in score_rows):
        raise ValueError('Outer heldout task found in capability evidence')
    jobs=unique_jobs(manifest,model,max_pixels)
    textjobs={j['task']:j for j in jobs if j['condition']=='text_only'}
    task_evidence={}
    provenance={}
    for task in sources:
        job=textjobs.get(task)
        if job is None:raise ValueError('Source task missing text-only evidence: '+task)
        record=records.get(job['key'])
        if not record or record.get('parsed') is None or any(
            record.get(field)!=job[field] for field in ('condition','task','model','prompt','max_pixels')):
            raise ValueError('Missing/incorrect source task-only evidence: '+task)
        task_evidence[task]=record['parsed']
        provenance[task]={'job_key':job['key'],'version':job['version']}
    profiles=build_expert_cue_proxy(sources,score_rows,task_evidence)
    return {'version':'ktrans-source-only-expert-cue-proxy-v1',
            'outer_task_excluded':outer,'source_tasks':sources,
            'k4_checkpoint':str(Path(manifest['k4_checkpoint']).resolve()),
            'task_text_cache_keys':provenance,'expert_profiles':profiles,
            'n_inner_source_pairs':len(score_rows),
            'evaluation_performed':False,
            'warning':'Descriptive source-only pseudo-task association, NOT measured causal/expert visual sensitivity. Do not use outer label metrics for selection.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--k5a-protocol',type=Path,required=True)
    p.add_argument('--pseudo-scores',type=Path,help='Default: sibling pseudo_transfer_scores.csv of K5-A protocol')
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    info=json.loads(args.k5a_protocol.read_text())
    scores=args.pseudo_scores or args.k5a_protocol.parent/'pseudo_transfer_scores.csv'
    result=prepare(manifest,info,read_transfer_scores(scores),
                   merge_records([args.reuse_cache,args.cache]),
                   args.model,args.max_pixels)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print('[alignment] sources=',len(result['source_tasks']),
          'nested_pairs=',result['n_inner_source_pairs'],'outer_task_excluded=',
          result['outer_task_excluded'],flush=True)
    for source,data in result['expert_profiles'].items():
        print(f'[alignment] {source} n_pairs={data["n_inner_pairs"]} '
              f'positive_weight={data["positive_transfer_weight"]:.3f}',flush=True)
    print('[alignment] saved',args.output,flush=True)
if __name__=='__main__':main()
