"""K7 pilot: frozen Qwen3-VL-8B image+task visual evidence, resumable JSONL.

No urban indicator labels are read, generated, or used for selection.
For each tile, cache image-only and image+task prompted results to
measure task-conditioned visual semantics before changing the expert bank.
"""
import argparse,hashlib,json,os,random,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
from PIL import Image
from urban_transfer.task_profiles import task_texts
from urban_transfer.protocol import SOURCE_TASKS

MODEL='Qwen/Qwen3-VL-8B-Instruct'
CUES=('built_up','roads','residential','vegetation','large_facilities','industrial_commercial')
TASKS=('education_accessibility','nighttime_light_intensity','digital_total_cell_intensity')
GENERAL='Describe visible urban morphology without guessing any measured indicator.'
SYSTEM=(
 'You are a conservative satellite imagery analyst. Use ONLY things visible in the '
 'supplied image. Do not infer measured accessibility, population counts, nightlight '
 'radiance, base station counts, presence of schools, or any numerical urban target. '
 'Do not assume a visible building is a school, hospital or telecom tower. '
 'Return ONLY a compact JSON object, with no markdown.'
)

def prompt(task,descriptions):
    header=(GENERAL if task=='image_only'
      else 'Task context (NOT a label or observed value): '+descriptions[task]+
           '\nDescribe observable scene evidence that may be relevant, but do not predict the indicator.')
    return (header+'\nReturn a JSON object with keys: "scene" (at most 40 words), '
      '"visual_cues" (object mapping each cue to ordinal 0,1,2,3 or null), '
      '"task_relevance" (object mapping each cue to ordinal 0,1,2,3 or null), '
      '"limitations" (short string). Cues: '+', '.join(CUES)+
      '. Scores: 0=not apparent/not relevant, 1=low, 2=moderate, 3=high; '
      'null=not assessable. For image_only, task_relevance must be null for all cues. '
      'Never output an indicator estimate, numerical count or measured intensity.')

def parse_output(raw):
    """Conservative parsing; retain raw for audit and do not invent numbers."""
    try:
        x=json.loads(raw)
    except json.JSONDecodeError:
        match=re.search(r'\{.*\}',raw,re.S)
        if not match:return None
        try:x=json.loads(match.group(0))
        except json.JSONDecodeError:return None
    if not isinstance(x,dict):return None
    for field in ('visual_cues','task_relevance'):
        m=x.get(field)
        if not isinstance(m,dict):return None
        for cue in CUES:
            val=m.get(cue)
            if val is not None and (type(val) is not int or val<0 or val>3):
                return None
    return {'scene':str(x.get('scene',''))[:500],
            'visual_cues':{c:x['visual_cues'].get(c) for c in CUES},
            'task_relevance':{c:x['task_relevance'].get(c) for c in CUES},
            'limitations':str(x.get('limitations',''))[:500]}

def locate_image(root,row):
    root=Path(root)
    image=Path(str(row.image_name))
    if image.is_absolute():
        raise ValueError('Absolute image name in dataset: inspect provenance')
    cities=[str(row.city)]
    if hasattr(row,'city_query') and pd.notna(row.city_query):
        cities.append(str(row.city_query))
    candidates=[root/image]
    for city in dict.fromkeys(cities):
        candidates.extend((root/city/image,root/city/'arcgis-imagery-FUA'/image))
    for path in candidates:
        if path.is_file():return path.resolve()
    raise FileNotFoundError(f'Cannot resolve tile city={row.city} image={image}')

def select_tiles(csv_path,image_root,n_tiles,seed):
    # Explicit usecols ensures that no target labels enter this experiment.
    cols=pd.read_csv(csv_path,nrows=0).columns.tolist()
    needed={'city','image_name'}
    if not needed.issubset(cols):raise ValueError('CSV must have city and image_name')
    usecols=['city','image_name']+(['city_query'] if 'city_query' in cols else [])
    frame=pd.read_csv(csv_path,usecols=usecols).dropna(subset=['city','image_name'])
    frame=frame.drop_duplicates(subset=['city','image_name']).reset_index(drop=True)
    if len(frame)<n_tiles:raise ValueError(f'Only {len(frame)} unique city/image tile identities')
    rng=random.Random(seed)
    grouped={city:list(grp.index) for city,grp in frame.groupby('city',sort=True)}
    for items in grouped.values():rng.shuffle(items)
    cities=sorted(grouped);rng.shuffle(cities)
    chosen=[]
    while len(chosen)<n_tiles:
        progress=False
        for city in cities:
            if grouped[city]:
                chosen.append(grouped[city].pop())
                progress=True
                if len(chosen)==n_tiles:break
        if not progress:break
    tiles=[]
    for idx in chosen:
        row=frame.iloc[idx]
        path=locate_image(image_root,row)
        tiles.append({'city':str(row.city),'image_name':str(row.image_name),
            'image_path':str(path),'tile_id':hashlib.sha256(str(path).encode()).hexdigest()[:20]})
    return tiles

def cache_key(tile_id,task,model,prompt_text,max_pixels):
    raw=json.dumps([tile_id,task,model,prompt_text,max_pixels],ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()

def run(args):
    if args.n_tiles<1 or args.max_new_tokens<32 or args.max_pixels<224*224:
        raise ValueError('Invalid tile/token/pixel limits')
    if not torch.cuda.is_available():
        raise RuntimeError('ROCm HIP device unavailable. Run on MI250X GPU allocation.')
    print('[k7] torch=',torch.__version__,'hip=',torch.version.hip,
          'gpu=',torch.cuda.get_device_name(0),flush=True)
    tasks=TASKS if args.tasks=='default' else tuple(args.tasks.split(','))
    if any(t not in SOURCE_TASKS for t in tasks):
        raise ValueError('Pilot only allows the six source task definitions')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    meta=checkpoint['meta']
    if meta.get('task_profile')!='structured':
        raise ValueError('K7 pilot expects K4 structured task text')
    descriptions=task_texts('structured')
    task_list=('image_only',)+tuple(tasks)
    tiles=select_tiles(meta['train_csv'],args.image_root,args.n_tiles,args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    previous={}
    if args.output.exists():
        with args.output.open() as f:
            for line in f:
                if not line.strip():continue
                rec=json.loads(line)
                previous[rec['key']]=rec
    jobs=[]
    for tile in tiles:
        for task in task_list:
            sentence=prompt(task,descriptions)
            key=cache_key(tile['tile_id'],task,args.model,sentence,args.max_pixels)
            if key in previous and previous[key].get('parsed') is not None:
                continue
            jobs.append((key,tile,task,sentence))
    print(f'[k7] tiles={len(tiles)} conditions={len(task_list)} '
          f'cached={len(previous)} pending={len(jobs)}',flush=True)
    if not jobs:return
    # Delay heavy imports/model download until after cache checks.
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    processor=AutoProcessor.from_pretrained(args.model,
                min_pixels=224*224,max_pixels=args.max_pixels)
    model=Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,dtype=torch.bfloat16,device_map='auto',
            attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
    for counter,(key,tile,task,sentence) in enumerate(jobs,1):
        with Image.open(tile['image_path']) as img:
            image=img.convert('RGB')
            messages=[
              {'role':'system','content':[{'type':'text','text':SYSTEM}]},
              {'role':'user','content':[{'type':'image','image':image},
                                       {'type':'text','text':sentence}]}]
            inputs=processor.apply_chat_template(messages,tokenize=True,
                        add_generation_prompt=True,return_dict=True,
                        return_tensors='pt').to(model.device)
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=args.max_new_tokens,
                             do_sample=False)
        trimmed=generated[0,inputs['input_ids'].shape[1]:]
        raw=processor.decode(trimmed,skip_special_tokens=True).strip()
        parsed=parse_output(raw)
        record={'key':key,'tile_id':tile['tile_id'],'city':tile['city'],
                'image_name':tile['image_name'],'image_path':tile['image_path'],
                'task':task,'model':args.model,'prompt':sentence,
                'max_pixels':args.max_pixels,'raw':raw,'parsed':parsed}
        with args.output.open('a') as f:
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
            f.flush()
            os.fsync(f.fileno())
        print(f'[k7] {counter}/{len(jobs)} task={task:<30} '
              f'city={tile["city"]} json={"OK" if parsed else "INVALID"}',flush=True)
        del inputs,generated,trimmed
    print('[k7] saved',args.output,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    p.add_argument('--output',type=Path,default=Path('runs/k7_pilot/vlm_evidence.jsonl'))
    p.add_argument('--model',default=MODEL)
    p.add_argument('--tasks',default='default',
                   help='Comma-separated subset of six source task IDs, or default for three pilot tasks')
    p.add_argument('--n-tiles',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-pixels',type=int,default=512*512)
    p.add_argument('--max-new-tokens',type=int,default=320)
    args=p.parse_args();run(args)

if __name__=='__main__':main()
