"""K7-D stage 2: batch/cache Qwen3-VL-8B knowledge for manifest rows.

Uses the unchanged K7-B task descriptions/prompt/rubric. A job is defined
only by image path and task text, NOT a target value. Reuses valid K7-B
cache records, and appends new inferences to a separate JSONL file.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from scripts.run_ktrans_k7_vlm_pilot import MODEL
from scripts.run_ktrans_k7b_controls import (
    SYSTEM,TEXT_ONLY_SYSTEM,VERSION,build_prompt,load_cached,parse)
from urban_transfer.task_profiles import task_texts

def unique_jobs(manifest,model,max_pixels):
    texts=task_texts('structured')
    tile_map={}
    image_task={}
    tasks=set()
    for row in manifest['rows']:
        tile=row['tile_id']
        if tile in tile_map and tile_map[tile]!=row['image_path']:
            raise ValueError('Tile hash collision or inconsistent image paths')
        tile_map[tile]=row['image_path']
        tasks.add(row['task'])
        image_task[(tile,row['task'])]=True
    jobs=[]
    def add(tile,task,condition,path):
        prompt=build_prompt(task,condition,texts)
        identity={'version':VERSION,'tile_id':tile,'task':task,'condition':condition,
                  'model':model,'prompt':prompt,'max_pixels':max_pixels}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        jobs.append({'key':key,'tile_id':tile,'task':task,'condition':condition,
                     'image_path':path,'prompt':prompt,
                     'model':model,'max_pixels':max_pixels})
    for tile,path in sorted(tile_map.items()):
        add(tile,'image_only','image_only',path)
    for tile,task in sorted(image_task):
        add(tile,task,'image_task',tile_map[tile])
    for task in sorted(tasks):
        add(None,task,'text_only',None)
    return jobs

def merge_records(paths):
    """Latest successful record wins; a later failure must not hide success."""
    records={}
    for path in paths:
        for row in load_cached(path).values():
            k=row['key']
            if row.get('parsed') is not None or k not in records:
                records[k]=row
    return records

def get_pending(jobs,records):
    return [j for j in jobs if not (j['key'] in records and records[j['key']].get('parsed') is not None)]

def messages_for(job):
    system=TEXT_ONLY_SYSTEM if job['condition']=='text_only' else SYSTEM
    content=[]
    if job['image_path'] is not None:
        with Image.open(job['image_path']) as img:
            content.append({'type':'image','image':img.convert('RGB')})
    content.append({'type':'text','text':job['prompt']})
    return [{'role':'system','content':[{'type':'text','text':system}]},
            {'role':'user','content':content}]

def infer_batch(model,processor,jobs,max_new_tokens):
    inputs=processor.apply_chat_template(
        [messages_for(job) for job in jobs],tokenize=True,
        add_generation_prompt=True,padding=True,
        return_dict=True,return_tensors='pt').to(model.device)
    with torch.inference_mode():
        tokens=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
    if tokens.shape[0]!=len(jobs):
        raise AssertionError('Batched output count mismatched image/task job count')
    return [x.strip() for x in processor.batch_decode(
        tokens[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',type=Path,default=Path('runs/k7d/evidence.jsonl'))
    p.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--model',default=MODEL)
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--max-new-tokens',type=int,default=440)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--dry-run',action='store_true',help='Print VLM cache coverage without loading the 8B model')
    p.add_argument('--max-new-jobs',type=int,default=0,
                   help='0=all outstanding; positive limit for incremental smoke tests')
    args=p.parse_args()
    if args.batch_size<1 or args.max_new_jobs<0 or args.max_new_tokens<64 or args.max_pixels<224*224:
        p.error('Invalid inference settings')
    manifest=json.loads(args.manifest.read_text())
    if manifest.get('version')!='k7d-manifest-v1':p.error('Unsupported manifest')
    jobs=unique_jobs(manifest,args.model,args.max_pixels)
    sources=[args.reuse_cache,args.output] if args.reuse_cache!=args.output else [args.output]
    records=merge_records(sources)
    pending=get_pending(jobs,records)
    print(f'[k7d-cache] total={len(jobs)} cached_valid={len(jobs)-len(pending)} '
          f'pending={len(pending)}',flush=True)
    if args.dry_run:return
    if args.max_new_jobs:pending=pending[:args.max_new_jobs]
    if not pending:return
    if not torch.cuda.is_available():
        raise RuntimeError('Qwen3-VL-8B inference requires an allocated ROCm GPU')
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    processor=AutoProcessor.from_pretrained(args.model,min_pixels=224*224,
                                             max_pixels=args.max_pixels)
    processor.tokenizer.padding_side='left'
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.model,dtype=torch.bfloat16,
          device_map='auto',attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    done=0;size=args.batch_size;begin=time.monotonic()
    # Do not mix no-image messages and image-bearing messages in one
    # multimodal processor call; some versions require homogeneous inputs.
    pending=sorted(pending,key=lambda x:(x['condition']=='text_only',x['key']))
    while done<len(pending):
        first=pending[done]
        group=[]
        for job in pending[done:done+size]:
            if (job['image_path'] is None)!=(first['image_path'] is None):break
            group.append(job)
        try:
            raws=infer_batch(model,processor,group,args.max_new_tokens)
        except (RuntimeError,torch.OutOfMemoryError) as exc:
            msg=str(exc).lower()
            oom=isinstance(exc,torch.OutOfMemoryError) or ('out of memory' in msg or 'memory allocation' in msg)
            if not oom or len(group)==1:raise
            size=max(1,len(group)//2)
            torch.cuda.empty_cache()
            print(f'[k7d-cache] OOM retry batch={size}',flush=True)
            continue
        with args.output.open('a') as f:
            for job,raw in zip(group,raws):
                record={**job,'version':VERSION,
                    'system_prompt':TEXT_ONLY_SYSTEM if job['condition']=='text_only' else SYSTEM,
                    'raw':raw,'parsed':parse(raw,job['condition'])}
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
                done+=1
                print(f'[k7d-cache] {done}/{len(pending)} {job["condition"]} '
                      f'{job["task"]} json={"OK" if record["parsed"] else "INVALID"}',flush=True)
            f.flush();os.fsync(f.fileno())
        print(f'[k7d-cache] batch={len(group)} elapsed_s={time.monotonic()-begin:.1f}',flush=True)
    print('[k7d-cache] saved',args.output,flush=True)

if __name__=='__main__':main()
