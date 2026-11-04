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
from urban_transfer.ktrans_k7d_image_only import (
    IMAGE_ONLY_VERSION,IMAGE_ONLY_SYSTEM,IMAGE_ONLY_PROMPT,parse_image_only)

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
        prompt=(IMAGE_ONLY_PROMPT if condition=='image_only' else
                build_prompt(task,condition,texts))
        version=IMAGE_ONLY_VERSION if condition=='image_only' else VERSION
        identity={'version':version,'tile_id':tile,'task':task,'condition':condition,
                  'model':model,'prompt':prompt,'max_pixels':max_pixels}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        jobs.append({'key':key,'version':version,'tile_id':tile,'task':task,'condition':condition,
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
        if not Path(path).is_file():continue
        with Path(path).open() as stream:
            for line in stream:
                if not line.strip():continue
                try:row=json.loads(line)
                except json.JSONDecodeError:continue  # interrupted final append
                if not isinstance(row,dict) or not isinstance(row.get('key'),str):
                    continue
                k=row['key']
                # Always favor the latest successful record, even if a
                # subsequent aborted attempt appended INVALID for that key.
                if row.get('parsed') is not None or k not in records:
                    records[k]=row
    return records

def get_pending(jobs,records):
    def valid(job):
        row=records.get(job['key'])
        return bool(row and row.get('parsed') is not None and all(
            row.get(k)==job[k] for k in
            ('version','tile_id','task','condition','model','max_pixels','prompt')))
    return [j for j in jobs if not valid(j)]

def messages_for(job):
    system=(IMAGE_ONLY_SYSTEM if job['condition']=='image_only' else
            TEXT_ONLY_SYSTEM if job['condition']=='text_only' else SYSTEM)
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
    p.add_argument('--extra-reuse-cache',type=Path,action='append',default=[],
                   help='Additional read-only JSONL knowledge caches; repeatable')
    p.add_argument('--model',default=MODEL)
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--max-new-tokens',type=int,default=440)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--dry-run',action='store_true',help='Print VLM cache coverage without loading the 8B model')
    p.add_argument('--conditions',default='image_only,image_task,text_only',
                   help='Comma-separated subset of cache conditions')
    p.add_argument('--max-new-jobs',type=int,default=0,
                   help='0=all outstanding; positive limit for incremental smoke tests')
    args=p.parse_args()
    if args.batch_size<1 or args.max_new_jobs<0 or args.max_new_tokens<64 or args.max_pixels<224*224:
        p.error('Invalid inference settings')
    manifest=json.loads(args.manifest.read_text())
    if manifest.get('version')!='k7d-manifest-v1':p.error('Unsupported manifest')
    conditions=set(x.strip() for x in args.conditions.split(',') if x.strip())
    if not conditions or not conditions.issubset({'image_only','image_task','text_only'}):
        p.error('Unknown cache condition')
    jobs=[j for j in unique_jobs(manifest,args.model,args.max_pixels)
          if j['condition'] in conditions]
    sources=list(dict.fromkeys([args.reuse_cache,*args.extra_reuse_cache,args.output]))
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
    done=0;size=args.batch_size;begin=time.monotonic();previous=time.monotonic()
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
                mode=job['condition']
                system=(IMAGE_ONLY_SYSTEM if mode=='image_only' else
                        TEXT_ONLY_SYSTEM if mode=='text_only' else SYSTEM)
                parsed=(parse_image_only(raw) if mode=='image_only' else parse(raw,mode))
                record={**job,'system_prompt':system,'raw':raw,'parsed':parsed}
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
                done+=1
                print(f'[k7d-cache] {done}/{len(pending)} {job["condition"]} '
                      f'{job["task"]} json={"OK" if record["parsed"] else "INVALID"}',flush=True)
            f.flush();os.fsync(f.fileno())
        now=time.monotonic()
        print(f'[k7d-cache] batch={len(group)} batch_s={now-previous:.1f} '
              f'total_elapsed_s={now-begin:.1f} rate={len(group)/max(now-previous,1e-6):.3f}/s',flush=True)
        previous=now
    print('[k7d-cache] saved',args.output,flush=True)

if __name__=='__main__':main()
