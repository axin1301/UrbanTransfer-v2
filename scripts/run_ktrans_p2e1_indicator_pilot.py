"""P2-E1 expert-aware indicator-definition/relation/region VLM pilot.

--dry-run builds a deterministic source-only sample/prompt manifest and
audits unknown units WITHOUT loading the VLM. No target labels influence
pilot selection or prompt. GPU mode is resumable, append-only. Strict JSON
parsing, and source provenance preserved for all cached outputs.
"""
import argparse,csv,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from urban_transfer.ktrans_p2e1_indicator_spec import catalog
from urban_transfer.ktrans_p2e1_pilot import (
    SYSTEM,VERSION,make_jobs,parse,sample_source_rows,descriptive_audit)

def cache_records(path):
    data={}
    if not path.is_file():return data
    with path.open() as f:
        for line in f:
            if not line.strip():continue
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if not isinstance(r,dict) or 'key' not in r:continue
            if r.get('parsed') is not None or r['key'] not in data:data[r['key']]=r
    return data

def is_valid(job,row):
    return bool(row and row.get('parsed') is not None and all(
        row.get(k)==job[k] for k in
        ('version','arm','task','tile_id','model','max_pixels','prompt')))

def infer(model,processor,job,max_new_tokens):
    from PIL import Image
    if job['image_path'] is None:
        content=[{'type':'text','text':job['prompt']}]
    else:
        with Image.open(job['image_path']) as im:
            content=[{'type':'image','image':im.convert('RGB')},
                     {'type':'text','text':job['prompt']}]
    msgs=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},
          {'role':'user','content':content}]
    inputs=processor.apply_chat_template([msgs],tokenize=True,
        add_generation_prompt=True,padding=True,
        return_dict=True,return_tensors='pt').to(model.device)
    import torch
    with torch.inference_mode():
        tokens=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
    return processor.batch_decode(tokens[:,inputs['input_ids'].shape[1]:],
                                  skip_special_tokens=True)[0].strip()

def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def freeze_manifest(path,manifest):
    """Compare canonical JSON objects rather than Python tuple/list containers.

    JSON turns source_tasks tuples into lists on disk. The exact same
    dry-run must therefore be accepted on a subsequent GPU resume,
    while any genuine prompt, sample, model, or metadata change
    remains a hard error rather than silently reusing stale results.
    """
    normalized=json.loads(json.dumps(manifest,ensure_ascii=False))
    path=Path(path)
    if path.exists():
        prior=json.loads(path.read_text())
        if prior!=normalized:
            raise ValueError('Existing pilot manifest differs: use a new output directory')
    else:
        path.write_text(json.dumps(normalized,indent=2,ensure_ascii=False))


def run(a):
    original=json.loads((a.input/'protocol.json').read_text())
    if original.get('version')!='ktrans-empirical-competence-v1':
        raise ValueError('Expected original source pseudo-task evidence')
    sources=tuple(sorted(original['source_tasks']))
    outer=original['outer_target_excluded']
    if outer in sources or len(sources)!=5:raise ValueError('Source-only five pseudo-tasks required')
    rows=[]
    with (a.input/'competence_rows.jsonl').open() as f:
        for line in f:
            if line.strip():rows.append(json.loads(line))
    if len(rows)!=int(original['n_expert_rows']):raise ValueError('Competence count mismatch')
    if any(row['pseudo_task'] not in sources or row['expert'] not in set(sources)-{row['pseudo_task']}
           for row in rows):raise ValueError('Outer target/own expert leaked')
    spec=catalog(a.reviewed_methodology)
    tiles=sample_source_rows(rows,sources,a.tiles_per_task_split,a.seed)
    jobs=make_jobs(tiles,sources,spec,a.model,a.max_pixels)
    a.output.mkdir(parents=True,exist_ok=True)
    plan={'version':VERSION,'outer_target_excluded':outer,
          'source_tasks':sources,'source_evidence':str(a.input.resolve()),
          'sample_rule':'SHA256(seed,source_task,SCUI_or_UCUI,tile_id), no error/label filtering',
          'tiles_per_task_split':a.tiles_per_task_split,'seed':a.seed,
          'n_tiles':len(tiles),'n_jobs':len(jobs),
          'max_pixels':a.max_pixels,'model':a.model,
          'reviewed_methodology':str(a.reviewed_methodology.resolve()) if a.reviewed_methodology else None,
          'unit_policy':'UNVERIFIED unless supplied in complete, source-referenced reviewed methodology JSON',
          'arms':['definition_only','definition_relation','definition_relation_region',
                  'permuted_relation_region'],
          'pilot_only':'Descriptive VLM output audit only. No target labels, expert losses or UCUI model selection in VLM prompts.',
          'warning':'DEV source evidence from Education-meta-selected K4 best.pt. SCUI/UCUI previously inspected; NOT independent zero-shot validation.'}
    # Freeze pilot manifest for reproducibility; refuse silent accidental resampling.
    manifest_path=a.output/'pilot_manifest.json'
    manifest={'plan':plan,'tiles':tiles,'jobs':jobs,'indicator_specs':spec}
    freeze_manifest(manifest_path,manifest)
    pending=[]
    cache=cache_records(a.output/'vlm_evidence.jsonl')
    for job in jobs:
        row=cache.get(job['key'])
        if row is not None and is_valid(job,row) and parse(row['raw'],sources,job['task'],
                job['image_path'] is not None) is not None:continue
        pending.append(job)
    print(f'[p2e1] source_tasks={len(sources)} sampled_tiles={len(tiles)} '
          f'jobs={len(jobs)} valid_cached={len(jobs)-len(pending)} pending={len(pending)}',flush=True)
    unknown=sum(s['raw_unit']=='unverified' for s in spec.values())
    print(f'[p2e1] unverified_raw_units={unknown}/{len(spec)} '
          f'new_gpu_jobs={min(len(pending),a.max_new_jobs) if a.max_new_jobs else len(pending)}',flush=True)
    if a.dry_run:return
    if pending:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('P2-E1 new multimodal inference requires ROCm GPU; dry-run runs on CPU')
        from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
        processor=AutoProcessor.from_pretrained(a.model,min_pixels=224*224,max_pixels=a.max_pixels)
        processor.tokenizer.padding_side='left'
        model=Qwen3VLForConditionalGeneration.from_pretrained(
            a.model,dtype=torch.bfloat16,device_map='auto',
            attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
        to_run=pending[:a.max_new_jobs] if a.max_new_jobs else pending
        with (a.output/'vlm_evidence.jsonl').open('a') as f:
            for i,job in enumerate(to_run,1):
                raw=infer(model,processor,job,a.max_new_tokens)
                parsed=parse(raw,sources,job['task'],job['image_path'] is not None)
                rec={**job,'system_prompt':SYSTEM,'raw':raw,'parsed':parsed}
                f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                f.flush();os.fsync(f.fileno())
                print(f'[p2e1] {i}/{len(to_run)} {job["arm"]} task={job["task"]} '
                      f'parsed={"OK" if parsed is not None else "INVALID"}',flush=True)
    cache=cache_records(a.output/'vlm_evidence.jsonl')
    valid=[]
    for job in jobs:
        row=cache.get(job['key'])
        if is_valid(job,row):
            parsed=parse(row['raw'],sources,job['task'],job['image_path'] is not None)
            if parsed is not None:valid.append({**row,'parsed':parsed})
    write_csv(a.output/'vlm_expert_hypotheses.csv',descriptive_audit(valid,sources))
    (a.output/'protocol.json').write_text(json.dumps({
        **plan,'n_valid_outputs':len(valid),
        'n_missing_or_invalid_outputs':len(jobs)-len(valid),
        'result_type':'VLM hypothesized expert compatibility; NOT empirical competence or measured expert accuracy'
    },indent=2))
    print(f'[p2e1] validated={len(valid)}/{len(jobs)} '
          f'saved={a.output}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('runs/ktrans_competence/education_all_outer_train_dev'))
    p.add_argument('--output',type=Path,default=Path('runs/ktrans_competence/p2e1_indicator_relation_source_dev'))
    p.add_argument('--reviewed-methodology',type=Path)
    p.add_argument('--tiles-per-task-split',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--max-new-tokens',type=int,default=750)
    p.add_argument('--max-new-jobs',type=int,default=0)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.tiles_per_task_split<1 or a.max_new_jobs<0 or a.max_new_tokens<128 or a.max_pixels<224*224:
        p.error('Invalid pilot budget')
    run(a)

if __name__=='__main__':main()
