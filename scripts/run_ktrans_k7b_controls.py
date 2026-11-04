"""K7-B: controlled 8B VLM visual proxy relevance experiment.

Three factorial conditions:
  image_task: same image, different tasks AND same task, different images
  text_only: same exact task prompt with no image (prompt-only control)
  image_only: image without task (scene-cue stability control)
Inputs are image path metadata from the existing K7-A cache. NO labels.
Inference is deterministic, append-only and resumable by prompt/model/condition.
"""
import argparse,hashlib,json,os,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from urban_transfer.task_profiles import task_texts
from scripts.run_ktrans_k7_vlm_pilot import MODEL,CUES,TASKS

VERSION='k7b-proxy-v1'
SYSTEM=(
    'You analyze overhead satellite images for observable urban morphology. '
    'Separate what is visually present from what might be an INDIRECT proxy '
    'for a named indicator. Do NOT predict indicator measurements or counts. '
    'Do NOT claim a building is a school, hospital or cellular station from appearance. '
    'A cue may be a plausible indirect proxy even when the target value is unknowable. '
    'Return a JSON object only, never markdown.'
)
SCORING=(
    'The six cue names are: '+', '.join(CUES)+'. '
    'Return a JSON object with fields: "scene" (one sentence), "visual_cues", '
    '"evidence_relevance", "evidence_confidence", "proxy_reasoning", "limitations". '
    'The three numerical dictionaries must have all six cue keys. '
    'visual_cues: 0=absent,1=weak,2=moderate,3=strong, null=not visually assessable. '
    'evidence_relevance: 0=no plausible proxy,1=weak,2=moderate,3=strong, '
    'null=the conceptual cue/task relationship cannot be judged. '
    'evidence_confidence: 0=no visual support,1=weak,2=moderate,3=strong, '
    'null=not visually assessable. '
    'Relevance and confidence are DIFFERENT: high relevance does not mean '
    'the image proves the indicator value. '
    'proxy_reasoning: one short sentence about one concrete cue and why it is '
    'or is not a defensible indirect proxy. '
    'For no-image text_only condition set visual_cues and evidence_confidence '
    'to null for every cue; give general conceptual evidence_relevance '
    'without pretending to see an image. '
    'For task-free image_only condition set evidence_relevance to null '
    'for every cue and describe only visible evidence. '
    'Never output the true indicator, a predicted indicator, or a numerical estimate.'
)
def build_prompt(task,mode,descriptions):
    if mode=='image_only':
        prefix='No named indicator. Describe only scene morphology.'
    else:
        prefix='Indicator definition (NOT an observed value): '+descriptions[task]+\
            '\nEvaluate whether each cue could be an INDIRECT spatial proxy for this task. '
        if mode=='text_only':
            prefix+='There is NO image. You must not infer the presence of any scene feature. '
    return prefix+'\n'+SCORING

def parse(raw,mode):
    try: obj=json.loads(raw)
    except json.JSONDecodeError:
        found=re.search(r'\{.*\}',raw,re.S)
        if not found:return None
        try: obj=json.loads(found.group(0))
        except json.JSONDecodeError:return None
    if not isinstance(obj,dict):return None
    parsed={}
    for field in ('visual_cues','evidence_relevance','evidence_confidence'):
        values=obj.get(field)
        if not isinstance(values,dict):return None
        parsed[field]={}
        for cue in CUES:
            if cue not in values:return None
            value=values[cue]
            if value is not None and (type(value) is not int or not 0<=value<=3):
                return None
            parsed[field][cue]=value
    if mode=='text_only' and any(parsed[field][cue] is not None
               for field in ('visual_cues','evidence_confidence') for cue in CUES):
        return None
    if mode=='image_only' and any(v is not None for v in parsed['evidence_relevance'].values()):
        return None
    parsed['scene']=str(obj.get('scene',''))[:400]
    parsed['proxy_reasoning']=str(obj.get('proxy_reasoning',''))[:600]
    parsed['limitations']=str(obj.get('limitations',''))[:400]
    return parsed

def read_tile_manifest(path,max_tiles):
    """Reuse the *exact* K7-A tiles; never resample or access indicator CSV."""
    selected={}
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():continue
            row=json.loads(line)
            identity=row['tile_id']
            item={'tile_id':identity,'city':row['city'],
                  'image_path':row['image_path'],'image_name':row['image_name']}
            if identity in selected and selected[identity]!=item:
                raise ValueError('Inconsistent tile identity within K7-A cache')
            selected.setdefault(identity,item)
    tiles=list(selected.values())
    if not tiles:raise ValueError('No tiles in K7-A manifest')
    if max_tiles is not None:tiles=tiles[:max_tiles]
    for tile in tiles:
        if not Path(tile['image_path']).is_file():
            raise FileNotFoundError(tile['image_path'])
    return tiles

def jobs_for(tiles,tasks,model,max_pixels):
    texts=task_texts('structured')
    result=[]
    for tile in tiles:
        for mode,task in [('image_only','image_only')]+[('image_task',t) for t in tasks]:
            sentence=build_prompt(task,mode,texts)
            identity={'version':VERSION,'tile_id':tile['tile_id'],'task':task,
                      'condition':mode,'model':model,'prompt':sentence,
                      'max_pixels':max_pixels}
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            result.append((key,tile,task,mode,sentence))
    for task in tasks:
        mode='text_only'
        sentence=build_prompt(task,mode,texts)
        identity={'version':VERSION,'tile_id':None,'task':task,
                  'condition':mode,'model':model,'prompt':sentence,
                  'max_pixels':max_pixels}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        result.append((key,None,task,mode,sentence))
    return result

def load_cached(path):
    values={}
    if Path(path).exists():
        with Path(path).open() as f:
            for line in f:
                if not line.strip():continue
                try:r=json.loads(line)
                except json.JSONDecodeError:
                    # Interrupted final line is incomplete, not a valid cache.
                    continue
                values[r['key']]=r
    return values

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('runs/k7_pilot/vlm_evidence.jsonl'))
    p.add_argument('--output',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--max-tiles',type=int,default=8)
    p.add_argument('--model',default=MODEL)
    p.add_argument('--max-pixels',type=int,default=512*512)
    p.add_argument('--max-new-tokens',type=int,default=440)
    p.add_argument('--tasks',default=','.join(TASKS))
    args=p.parse_args()
    tasks=tuple(dict.fromkeys(x.strip() for x in args.tasks.split(',') if x.strip()))
    if not tasks or any(t not in task_texts('structured') for t in tasks):
        p.error('Unknown or empty task list')
    if args.max_tiles<1 or args.max_new_tokens<64 or args.max_pixels<224*224:
        p.error('Invalid sample/token/pixel limits')
    tiles=read_tile_manifest(args.manifest,args.max_tiles)
    jobs=jobs_for(tiles,tasks,args.model,args.max_pixels)
    cache=load_cached(args.output)
    remaining=[job for job in jobs if not (job[0] in cache and
       cache[job[0]].get('parsed') is not None)]
    print(f'[k7b] tiles={len(tiles)} tasks={len(tasks)} jobs={len(jobs)} '
          f'cached_success={len(jobs)-len(remaining)} pending={len(remaining)}',flush=True)
    if not remaining:return
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA/HIP device; use MI250X allocation')
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    processor=AutoProcessor.from_pretrained(args.model,min_pixels=224*224,
                                               max_pixels=args.max_pixels)
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.model,
           dtype=torch.bfloat16,device_map='auto',
           attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for k,(key,tile,task,mode,sentence) in enumerate(remaining,1):
        content=[]
        if tile is not None:
            with Image.open(tile['image_path']) as image:
                content.append({'type':'image','image':image.convert('RGB')})
                content.append({'type':'text','text':sentence})
                messages=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},
                          {'role':'user','content':content}]
                inputs=processor.apply_chat_template(messages,tokenize=True,
                         add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
        else:
            messages=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},
                      {'role':'user','content':[{'type':'text','text':sentence}]}]
            inputs=processor.apply_chat_template(messages,tokenize=True,
                     add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
        raw=processor.decode(generated[0,inputs['input_ids'].shape[1]:],
                             skip_special_tokens=True).strip()
        result={'key':key,'version':VERSION,'tile_id':tile['tile_id'] if tile else None,
                'city':tile['city'] if tile else None,
                'image_name':tile['image_name'] if tile else None,
                'task':task,'condition':mode,'model':args.model,'prompt':sentence,
                'max_pixels':args.max_pixels,'raw':raw,'parsed':parse(raw,mode)}
        with args.output.open('a') as f:
            f.write(json.dumps(result,ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno())
        print(f'[k7b] {k}/{len(remaining)} condition={mode} task={task} '
              f'json={"OK" if result["parsed"] else "INVALID"}',flush=True)
    print('[k7b] saved',args.output,flush=True)

if __name__=='__main__':main()
