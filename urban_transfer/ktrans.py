"""K0+K1: frozen RemoteCLIP text semantics and city/task meta-validation.

Independent from M2/M3. This is a research prototype, not the full K-TransMoE.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from .data_m1 import task_vector
from .protocol import TASK_DESCRIPTIONS
from .m3_features import FeatureStore, CachedFeatureDataset, FrozenRemoteCLIP
from .m3 import M3Predictor
from .ktrans_scale import TaskScalePredictor
from .ktrans_experts import SemanticExpertBank
from .ktrans_normalized import NormalizedExpertBank,source_moments
from .ktrans_text import qwen_task_vectors
from .model import ModelConfig
from .m2 import loop, seed_all
from .cache import HiddenStateCache
from .checkpoints import load_checkpoint, save_checkpoint
from .formatting import metric_text

class SemanticDataset(Dataset):
    def __init__(self,base,task_vectors):
        self.base=base
        self.df=base.df
        self.task_vectors=task_vectors
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        sample=self.base[i]
        sample['task']=self.task_vectors[sample['task_name']].clone()
        return sample

def task_embeddings(args):
    """Frozen RemoteCLIP text encoder; compute once and persist all task vectors."""
    root=Path(args.text_cache_dir);root.mkdir(parents=True,exist_ok=True)
    descriptions=[f'Urban geographic indicator: {TASK_DESCRIPTIONS[name]}' for name in sorted(TASK_DESCRIPTIONS)]
    tasks=sorted(TASK_DESCRIPTIONS)
    cache_meta={'encoder':'RemoteCLIP-'+args.remoteclip_model,
                'checkpoint':str(Path(args.remoteclip_checkpoint).resolve()) if args.remoteclip_checkpoint else 'official-hf',
                'tasks':list(zip(tasks,descriptions))}
    token=hashlib.sha256(json.dumps(cache_meta,sort_keys=True).encode()).hexdigest()[:16]
    path=root/f'task_text_{token}.pt'
    if path.exists():
        data=torch.load(path,map_location='cpu',weights_only=True)
        if set(data)!=set(tasks): raise ValueError('Task embedding cache mismatch')
        return data
    import open_clip
    encoder=FrozenRemoteCLIP(args.remoteclip_model,args.remoteclip_checkpoint,
                             device='cuda' if torch.cuda.is_available() else 'cpu')
    with torch.inference_mode():
        tokens=open_clip.tokenize(descriptions).to(encoder.device)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=encoder.device.type=='cuda'):
            vectors=encoder.model.encode_text(tokens).float()
        vectors=torch.nn.functional.normalize(vectors,dim=-1).cpu()
    result={task:vectors[i].clone() for i,task in enumerate(tasks)}
    torch.save(result,path)
    print(f'[k1] cached {len(tasks)} frozen text vectors to {path}',flush=True)
    return result

def meta_indices(df,heldout_task,city_fraction,seed):
    """City-disjoint meta validation plus held-out task; no heldout labels in training."""
    if heldout_task not in set(df.indicator_column):
        raise ValueError(f'Missing heldout task {heldout_task}')
    if not 0<city_fraction<0.5:raise ValueError('city fraction must be in (0,0.5)')
    cities=np.array(sorted(df.city.astype(str).unique()))
    if len(cities)<2:raise ValueError('Need >=2 source cities')
    np.random.default_rng(seed).shuffle(cities)
    n=max(1,min(len(cities)-1,round(len(cities)*city_fraction)))
    unseen=set(cities[:n])
    is_held=df.indicator_column.eq(heldout_task).to_numpy()
    is_new_city=df.city.astype(str).isin(unseen).to_numpy()
    split={
       'train':np.flatnonzero(~is_held & ~is_new_city).tolist(),
       'meta_SCUI':np.flatnonzero(is_held & ~is_new_city).tolist(),
       'meta_UCSI':np.flatnonzero(~is_held & is_new_city).tolist(),
       'meta_UCUI':np.flatnonzero(is_held & is_new_city).tolist(),
    }
    if any(not value for value in split.values()):
        raise ValueError('Empty meta partition; increase --limit')
    return split,sorted(unseen)

def dataset(args,task_vectors):
    store=FeatureStore(args.feature_dir,encoder_id='RemoteCLIP-'+args.remoteclip_model)
    dim=len(next(iter(task_vectors.values())))
    ds=CachedFeatureDataset(args.train_csv,args.image_root,store,partition='train',
         limit=args.limit,task_dim=dim,image_column=args.image_column,seed=args.seed)
    return SemanticDataset(ds,task_vectors)

def make_model(args,input_dim,task_dim,n_sources,task_vectors=None,source_tasks=None,source_stats=None):
    cfg=ModelConfig(visual_dim=args.visual_dim,task_dim=task_dim,hidden_dim=args.hidden_dim,
        n_experts=args.experts,top_k=args.top_k,n_sources=n_sources,use_knowledge=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if getattr(args,'expert_bank',False):
        if task_vectors is None or source_tasks is None:
            raise ValueError('Expert bank needs task vectors and source tasks')
        if getattr(args,'normalized_experts',False):
            if source_stats is None:raise ValueError('Normalized expert bank needs source-only training moments')
            model=NormalizedExpertBank(input_dim,task_vectors,source_tasks,source_stats,
                  hidden_dim=args.hidden_dim,temperature=args.router_temperature)
        else:
            model=SemanticExpertBank(input_dim,task_vectors,source_tasks,
                  hidden_dim=args.hidden_dim,temperature=args.router_temperature)
        return cfg,model.to(device),device
    model=(TaskScalePredictor(cfg,input_dim,task_dim) if getattr(args,'scale_head',False)
           else M3Predictor(cfg,input_dim))
    return cfg,model.to(device),device

def run(args):
    seed_all(args.seed)
    if args.task_encoder=='qwen3':
        task_vectors=qwen_task_vectors(root=args.text_cache_dir,model_id=args.text_model_id,
            profile=args.task_profile,llm_profile_file=args.llm_profile_file,only_cached=True)
    elif args.task_encoder=='remoteclip':
        task_vectors=task_embeddings(args)
    else:
        task_vectors={task:torch.from_numpy(task_vector(desc,128).copy())
                for task,desc in TASK_DESCRIPTIONS.items()}
    ds=dataset(args,task_vectors)
    split,unseen_cities=meta_indices(ds.df,args.heldout_task,args.meta_city_fraction,args.seed)
    print(f'[k0] sampled={len(ds)} source cities={ds.df.city.nunique()} '
          f'source tasks={ds.df.indicator_column.nunique()} '
          f'heldout_task={args.heldout_task} heldout_cities={len(unseen_cities)}',flush=True)
    print('[k0] '+ ' '.join(f'{name}={len(ids)}' for name,ids in split.items()),flush=True)
    in_dim=ds[0]['image'].numel()
    source_tasks=tuple(sorted(set(ds.df.iloc[split['train']].indicator_column)))
    if getattr(args,'normalized_experts',False) and not getattr(args,'expert_bank',False):
        raise ValueError('--normalized-experts requires --expert-bank')
    stats=(source_moments(ds.df.iloc[split['train']],source_tasks)
           if getattr(args,'normalized_experts',False) else None)
    cfg,model,device=make_model(args,in_dim,len(next(iter(task_vectors.values()))),
                len(source_tasks),task_vectors,source_tasks,stats)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    # No provider or Qwen; this mode only tests frozen visual/text embeddings.
    cache=HiddenStateCache(args.cache_dir,'unused',enabled=False)
    subsets={name:Subset(ds,idx) for name,idx in split.items()}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    meta={'pipeline':'ktrans-k0-k1','task_encoder':args.task_encoder,'heldout_task':args.heldout_task,
          'heldout_cities':unseen_cities,'seed':args.seed,'limit':args.limit,
          'feature_dir':args.feature_dir,'remoteclip_model':args.remoteclip_model,
          'train_csv':str(Path(args.train_csv).resolve()),'meta_city_fraction':args.meta_city_fraction,
          'scale_head':getattr(args,'scale_head',False),
          'expert_bank':getattr(args,'expert_bank',False),
          'router_temperature':getattr(args,'router_temperature',0.15),
          'source_tasks':list(source_tasks),
          'normalized_experts':getattr(args,'normalized_experts',False),
          'source_stats':stats,
          'text_model_id':getattr(args,'text_model_id',None),
          'task_profile':getattr(args,'task_profile','short'),
          'llm_profile_file':getattr(args,'llm_profile_file',None)}
    best=float('inf');best_epoch=0
    for epoch in range(1,args.epochs+1):
        tr=loop(model,subsets['train'],args,None,(),device,cache,optimizer=optimizer)
        vals={name:loop(model,subsets[name],args,None,(),device,cache,split_name=name)
              for name in ('meta_SCUI','meta_UCSI','meta_UCUI')}
        # Explicit development-set selection, not held-out test performance.
        score=np.mean([vals[name]['overall']['mse'] for name in ('meta_SCUI','meta_UCSI','meta_UCUI')])
        if score<best:best=score;best_epoch=epoch
        meta['best_epoch']=best_epoch
        save_checkpoint(out/'last.pt',model=model,optimizer=optimizer,cfg=cfg,meta=meta,epoch=epoch,best_val=best)
        if epoch==best_epoch:
            save_checkpoint(out/'best.pt',model=model,optimizer=optimizer,cfg=cfg,meta=meta,epoch=epoch,best_val=best)
        record={'epoch':epoch,'train':tr,'meta_validation':vals,'best_meta_mse':best}
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(f'[ktrans] epoch={epoch} train_mse={tr["mean_mse"]:.3f} '
              + ' '.join(f'{name}={vals[name]["overall"]["mse"]:.3f}' for name in vals)
              +f' best_meta={best:.3f}',flush=True)
        if args.patience>0 and epoch-best_epoch>=args.patience:break
    (out/'meta_protocol.json').write_text(json.dumps(meta,indent=2))
    print('[k1] best epoch',best_epoch,'selection meta MSE',f'{best:.3f}',flush=True)

def evaluate_meta(args):
    if not args.checkpoint:raise ValueError('--checkpoint is required')
    raw=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    meta=raw['meta']
    if meta.get('pipeline')!='ktrans-k0-k1':raise ValueError('Not K0/K1 checkpoint')
    for key in ('task_encoder','heldout_task','remoteclip_model','limit','seed','meta_city_fraction','scale_head'):
        if getattr(args,key)!=meta[key]:raise ValueError(f'Checkpoint config mismatch: {key}')
    if getattr(args,'normalized_experts',False)!=meta.get('normalized_experts',False):
        raise ValueError('Checkpoint normalized-experts mode mismatch')
    if getattr(args,'expert_bank',False)!=meta.get('expert_bank',False):
        raise ValueError('Checkpoint expert-bank mode mismatch')
    if getattr(args,'expert_bank',False) and args.router_temperature!=meta['router_temperature']:
        raise ValueError('Checkpoint router temperature mismatch')
    if args.task_encoder=='qwen3':
        if args.text_model_id!=meta['text_model_id'] or args.task_profile!=meta['task_profile']:
            raise ValueError('Qwen encoder/profile differs from checkpoint')
        vectors=qwen_task_vectors(root=args.text_cache_dir,model_id=args.text_model_id,
            profile=args.task_profile,llm_profile_file=args.llm_profile_file,only_cached=True)
    elif args.task_encoder=='remoteclip':
        vectors=task_embeddings(args)
    else:
        vectors={
            task:torch.from_numpy(task_vector(desc,128).copy()) for task,desc in TASK_DESCRIPTIONS.items()}
    ds=dataset(args,vectors)
    split,cities=meta_indices(ds.df,args.heldout_task,args.meta_city_fraction,args.seed)
    if cities!=meta['heldout_cities']:raise ValueError('Meta city split mismatch')
    config=raw['config']
    args.task_dim=config['task_dim'];args.visual_dim=config['visual_dim']
    args.hidden_dim=config['hidden_dim'];args.experts=config['n_experts'];args.top_k=config['top_k']
    cfg,model,device=make_model(args,ds[0]['image'].numel(),len(next(iter(vectors.values()))),
                                 config['n_sources'],vectors,meta.get('source_tasks'),meta.get('source_stats'))
    load_checkpoint(args.checkpoint,model=model,device=device)
    cache=HiddenStateCache(args.cache_dir,'unused',enabled=False)
    results={}
    for name in ('meta_SCUI','meta_UCSI','meta_UCUI'):
        results[name]=loop(model,Subset(ds,split[name]),args,None,(),device,cache,split_name=name)
        print(name,metric_text(results[name]['overall']),flush=True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'meta_metrics.json').write_text(json.dumps(results,indent=2))
    print('[k1] These are DEVELOPMENT meta-validation metrics, not unbiased final test results.')
