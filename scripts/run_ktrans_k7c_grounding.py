"""K7-C causal-input probes on the exact K7-B tiles and prompts.

Keep SYSTEM, prompt and model identical to the original image_task call.
Only modify the image pixels:
  blank: spatial content removed, same shape and constant RGB
  patch_shuffle: 4x4 deterministic grid permutation of the same pixels
No label columns, pseudo-labels, gradients or checkpoint updates.
The original image-task baseline is read from K7-B, never re-generated.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from scripts.run_ktrans_k7_vlm_pilot import MODEL,TASKS
from scripts.run_ktrans_k7b_controls import (
    SYSTEM,VERSION as B_VERSION,build_prompt,load_cached,parse,read_tile_manifest)
from urban_transfer.task_profiles import task_texts

VERSION='k7c-pixel-interventions-v1'
CONDITIONS=('blank','patch_shuffle')

def transform_image(image,condition,tile_id):
    """Explicit pixel intervention. Never permute tile identities or task labels."""
    rgb=image.convert('RGB')
    if condition=='blank':
        return Image.new('RGB',rgb.size,(128,128,128))
    if condition!='patch_shuffle':
        raise ValueError('Unknown image intervention')
    arr=np.array(rgb,dtype=np.uint8)
    h,w=arr.shape[:2]
    # Exact deterministic 4x4 permutation after resizing to 4-divisible dims.
    nh=max(4,(h//4)*4);nw=max(4,(w//4)*4)
    if (nh,nw)!=(h,w):
        arr=np.array(rgb.resize((nw,nh)),dtype=np.uint8)
    patch_h,patch_w=nh//4,nw//4
    patches=[arr[i*patch_h:(i+1)*patch_h,j*patch_w:(j+1)*patch_w,:].copy()
             for i in range(4) for j in range(4)]
    seed=int(hashlib.sha256((VERSION+tile_id).encode()).hexdigest()[:16],16)
    permutation=np.random.default_rng(seed).permutation(16)
    if np.array_equal(permutation,np.arange(16)):
        permutation=np.roll(permutation,1)
    out=np.empty_like(arr)
    for i in range(4):
        for j in range(4):
            out[i*patch_h:(i+1)*patch_h,j*patch_w:(j+1)*patch_w,:]=patches[permutation[4*i+j]]
    return Image.fromarray(out)

def jobs_for(tiles,tasks,model,max_pixels,conditions=CONDITIONS):
    texts=task_texts('structured')
    jobs=[]
    for tile in tiles:
        for task in tasks:
            sentence=build_prompt(task,'image_task',texts)
            for condition in conditions:
                identity={'version':VERSION,'tile_id':tile['tile_id'],
                    'task':task,'condition':condition,'model':model,
                    'prompt':sentence,'system':SYSTEM,'max_pixels':max_pixels}
                key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
                jobs.append((key,tile,task,condition,sentence))
    return jobs

def original_index(cache,tiles,tasks,model,max_pixels):
    """Verify baseline prompts/model/pixels match intervention requests exactly."""
    manifest={t['tile_id'] for t in tiles}
    records={}
    required_prompts=task_texts('structured')
    for row in load_cached(cache).values():
        if row.get('condition')!='image_task':continue
        target=(row.get('tile_id'),row.get('task'))
        if target[0] not in manifest or target[1] not in tasks:continue
        if row.get('model')!=model or row.get('max_pixels')!=max_pixels:
            continue
        if row.get('prompt')!=build_prompt(target[1],'image_task',required_prompts):
            continue
        if row.get('system_prompt',SYSTEM)!=SYSTEM:
            raise ValueError('Baseline system instruction differs from K7-C')
        if row.get('parsed') is None:
            raise ValueError('Original image_task JSON invalid for '+str(target))
        records[target]=row
    missing=[(tile['tile_id'],task) for tile in tiles for task in tasks
             if (tile['tile_id'],task) not in records]
    if missing:raise ValueError('Missing exact matched K7-B image_task baselines: '+str(missing[:5]))
    return records

def prepare_batch(jobs):
    conversations=[]
    for _,tile,_,condition,sentence in jobs:
        with Image.open(tile['image_path']) as image:
            altered=transform_image(image,condition,tile['tile_id'])
            conversations.append([
              {'role':'system','content':[{'type':'text','text':SYSTEM}]},
              {'role':'user','content':[{'type':'image','image':altered},
                                        {'type':'text','text':sentence}]}])
    return conversations

def decode_batch(processor,generated,input_ids):
    width=input_ids.shape[1]
    if generated.shape[0]!=input_ids.shape[0]:
        raise ValueError('Incorrect generation batch dimension')
    return [x.strip() for x in processor.batch_decode(
        generated[:,width:],skip_special_tokens=True)]

def run_batch(model,processor,jobs,max_new_tokens):
    inputs=processor.apply_chat_template(
        prepare_batch(jobs),tokenize=True,add_generation_prompt=True,
        padding=True,return_dict=True,return_tensors='pt').to(model.device)
    with torch.inference_mode():
        generated=model.generate(**inputs,max_new_tokens=max_new_tokens,
                                 do_sample=False)
    return decode_batch(processor,generated,inputs['input_ids'])

def is_oom(error):
    message=str(error).lower()
    return isinstance(error,torch.OutOfMemoryError) or (
        isinstance(error,RuntimeError) and
        ('out of memory' in message or 'memory allocation' in message))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('runs/k7_pilot/vlm_evidence.jsonl'))
    p.add_argument('--original',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--output',type=Path,default=Path('runs/k7c_pilot/interventions.jsonl'))
    p.add_argument('--max-tiles',type=int,default=8)
    p.add_argument('--tasks',default=','.join(TASKS))
    p.add_argument('--conditions',default=','.join(CONDITIONS))
    p.add_argument('--model',default=MODEL)
    p.add_argument('--max-pixels',type=int,default=512*512)
    p.add_argument('--max-new-tokens',type=int,default=440)
    p.add_argument('--batch-size',type=int,default=4)
    args=p.parse_args()
    tasks=tuple(dict.fromkeys(t.strip() for t in args.tasks.split(',') if t.strip()))
    conditions=tuple(dict.fromkeys(t.strip() for t in args.conditions.split(',') if t.strip()))
    if not tasks or not set(tasks).issubset(TASKS):
        p.error('Select one or more source pilot tasks')
    if not conditions or not set(conditions).issubset(CONDITIONS):
        p.error('Unknown interventions')
    if args.batch_size<1 or args.max_tiles<1 or args.max_pixels<224*224 or args.max_new_tokens<64:
        p.error('Invalid limits')
    tiles=read_tile_manifest(args.manifest,args.max_tiles)
    baselines=original_index(args.original,tiles,tasks,args.model,args.max_pixels)
    jobs=jobs_for(tiles,tasks,args.model,args.max_pixels,conditions)
    prev=load_cached(args.output)
    pending=[j for j in jobs if not (j[0] in prev and prev[j[0]].get('parsed') is not None)]
    print(f'[k7c] tiles={len(tiles)} tasks={len(tasks)} baselines={len(baselines)} '
          f'jobs={len(jobs)} cached={len(jobs)-len(pending)} pending={len(pending)}',flush=True)
    if not pending:return
    if not torch.cuda.is_available():raise RuntimeError('Requires ROCm/CUDA GPU for frozen 8B inference')
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    processor=AutoProcessor.from_pretrained(args.model,min_pixels=224*224,max_pixels=args.max_pixels)
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.model,dtype=torch.bfloat16,
          device_map='auto',attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for count,(key,tile,task,condition,sentence) in enumerate(pending,1):
        with Image.open(tile['image_path']) as image:
            altered=transform_image(image,condition,tile['tile_id'])
            messages=[
              {'role':'system','content':[{'type':'text','text':SYSTEM}]},
              {'role':'user','content':[{'type':'image','image':altered},
                                       {'type':'text','text':sentence}]}]
            inputs=processor.apply_chat_template(messages,tokenize=True,
                        add_generation_prompt=True,return_dict=True,
                        return_tensors='pt').to(model.device)
        with torch.inference_mode():
            tokens=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
        raw=processor.decode(tokens[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        parsed=parse(raw,'image_task')
        record={'key':key,'version':VERSION,'tile_id':tile['tile_id'],
          'city':tile['city'],'image_path':tile['image_path'],
          'image_name':tile['image_name'],'task':task,'condition':condition,
          'model':args.model,'max_pixels':args.max_pixels,'prompt':sentence,
          'system_prompt':SYSTEM,'raw':raw,'parsed':parsed}
        with args.output.open('a') as out:
            out.write(json.dumps(record,ensure_ascii=False)+'\n')
            out.flush();os.fsync(out.fileno())
        print(f'[k7c] {count}/{len(pending)} {tile["city"]} {task} '
              f'{condition} json={"OK" if parsed else "INVALID"}',flush=True)
    print('[k7c] saved',args.output,flush=True)

if __name__=='__main__':main()
