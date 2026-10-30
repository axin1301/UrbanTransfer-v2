"""M2 runner: separate training and evaluation, frozen MLLM cache and checkpoint resume."""
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from .data_m1 import ImageUrbanDataset
from .engine import Predictor,build_provider,resolve_batch_knowledge,metrics
from .model import ModelConfig
from .cache import HiddenStateCache
from .ablations import PermutedKnowledgeDataset
from .checkpoints import save_checkpoint,load_checkpoint
from .formatting import epoch_text,metric_text

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def data_from_path(path,args,partition):
    return ImageUrbanDataset(path,args.image_root,partition=partition,limit=args.limit,
        task_dim=args.task_dim,require_images=not args.allow_missing_images,
        image_column=args.image_column)

def source_info(df):
    tasks=tuple(sorted(set(df.indicator_column)))
    if not tasks:raise ValueError('No source tasks')
    return tasks

def make_components(args,source_tasks):
    cfg=ModelConfig(use_knowledge=(args.provider!='no-llm'),n_sources=len(source_tasks),
        task_dim=args.task_dim,visual_dim=args.visual_dim,hidden_dim=args.hidden_dim,
        n_experts=args.experts,top_k=args.top_k)
    provider=(None if args.provider=='no-llm' else build_provider('qwen3vl-latent' if args.provider=='shuffled-latent' else args.provider,args.model_id))
    latent_dim=provider.model.config.text_config.hidden_size if args.provider in ('qwen3vl-latent','shuffled-latent') else None
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.provider not in ('mock','no-llm') and device.type!='cuda':raise RuntimeError('Real Qwen3-VL requires GPU')
    model=Predictor(cfg,latent_dim=latent_dim).to(device)
    cache=HiddenStateCache(args.cache_dir,args.model_id,enabled=(args.provider in ('qwen3vl-latent','shuffled-latent') and not args.no_cache))
    return cfg,provider,model,device,cache

def obtain_knowledge(batch,args,provider,source_tasks,device,cache):
    if args.provider=='no-llm':
        return None
    if args.provider=='shuffled-latent':
        batch=dict(batch)
        batch['image_path']=batch['knowledge_image_path']
        batch['city']=batch['knowledge_city']
        batch['task_description']=batch['knowledge_task_description']
    if args.provider in ('qwen3vl-latent','shuffled-latent') and cache.enabled:
        result=cache.get_or_compute_many(provider=provider,
                requests=[dict(image_path=path,city=city,task_description=description,source_tasks=source_tasks)
                          for city,path,description in zip(batch['city'],batch['image_path'],batch['task_description'])])
        return torch.stack(result).to(device,dtype=torch.float32)
    return resolve_batch_knowledge(batch,provider,source_tasks,device,
                                   'qwen3vl-latent' if args.provider=='shuffled-latent' else args.provider)

def loop(model,ds,args,provider,source_tasks,device,cache,optimizer=None,split_name=None):
    train=optimizer is not None
    model.train(train)
    if args.provider=='shuffled-latent':
        ds=PermutedKnowledgeDataset(ds,seed=args.seed+({'train':0,'validation':1}.get(split_name,2)))
    eval_batch_size=getattr(args,'eval_batch_size',64)
    if eval_batch_size<1:raise ValueError('--eval-batch-size must be positive')
    workers=getattr(args,'eval_workers',0) if not train else 0
    loader=DataLoader(ds,batch_size=args.batch_size if train else eval_batch_size,
                      shuffle=train,num_workers=workers,pin_memory=(device.type=='cuda'))
    truth=[];prediction=[];groups={};loss_sum=0.
    done=0
    if split_name:
        print(f'[eval] {split_name}: {len(ds)} rows, provider={args.provider}, cache={cache.enabled}',flush=True)
    for batch in loader:
        batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        knowledge=obtain_knowledge(batch,args,provider,source_tasks,device,cache)
        with torch.set_grad_enabled(train):
            y=model(batch,knowledge)
            loss=torch.nn.functional.mse_loss(y,batch['target'])
            if train:
                optimizer.zero_grad();loss.backward();optimizer.step()
        count=len(batch['task_name']);loss_sum+=float(loss.item())*count
        done+=count
        if split_name and (done % max((args.batch_size if train else eval_batch_size)*10,1) < count or done==len(ds)):
            print(f'[eval] {split_name}: {done}/{len(ds)} rows',flush=True)
        if not train:
            for actual,pred,city,task in zip(batch['target'].detach().cpu().tolist(),
                  y.detach().cpu().tolist(),batch['city'],batch['task_name']):
                truth.append(actual);prediction.append(pred)
                key=(city,task);groups.setdefault(key,([],[]))
                groups[key][0].append(actual);groups[key][1].append(pred)
    if train:return {'mean_mse':loss_sum/max(len(ds),1),'n':len(ds)}
    per_city={};per_task={}
    for (city,task),(ys,ps) in groups.items():
        cy=per_city.setdefault(city,([],[]));cy[0].extend(ys);cy[1].extend(ps)
        tt=per_task.setdefault(task,([],[]));tt[0].extend(ys);tt[1].extend(ps)
    return {'overall':metrics(truth,prediction),
            'per_city':{k:metrics(*v) for k,v in per_city.items()},
            'per_task':{k:metrics(*v) for k,v in per_task.items()}}

def check_splits(train_df,others):
    train_cities=set(train_df.city);train_tasks=set(train_df.indicator_column)
    for name,df in others.items():
        if name in ('UCSI','UCUI') and train_cities.intersection(set(df.city)):
            raise ValueError(f'City leakage in {name}')
        if name in ('SCUI','UCUI') and train_tasks.intersection(set(df.indicator_column)):
            raise ValueError(f'Task leakage in {name}')

def get_meta(args,source_tasks):
    return {'provider':args.provider,'model_id':args.model_id,'source_tasks':list(source_tasks),
        'seed':args.seed,'train_csv':str(Path(args.train_csv).resolve()),
        'image_root':args.image_root,'image_column':args.image_column,
        'config_schema':'m2.2','cache_enabled':not args.no_cache,
        'val_fraction':args.val_fraction,'limit':args.limit}

def train(args):
    seed_all(args.seed)
    ds=data_from_path(args.train_csv,args,'train')
    source_tasks=source_info(ds.df)
    cfg,provider,model,device,cache=make_components(args,source_tasks)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    metadata=get_meta(args,source_tasks)
    start=0;best=float('inf');best_epoch=0
    if args.resume:
        ckpt=load_checkpoint(args.resume,model=model,optimizer=optimizer,restore_rng=True,device=device)
        if ckpt['meta']['source_tasks'] != list(source_tasks):
            raise ValueError('Cannot resume with different source tasks')
        if ckpt['meta']['provider'] != args.provider or ckpt['meta']['model_id'] != args.model_id:
            raise ValueError('Cannot resume with different provider/model')
        for key in ('seed','train_csv','val_fraction','limit'):
            if ckpt['meta'].get(key) != metadata.get(key):
                raise ValueError(f'Cannot resume: {key} changed')
        start=int(ckpt['epoch']);best=ckpt.get('best_val') if ckpt.get('best_val') is not None else float('inf')
        best_epoch=int(ckpt['meta'].get('best_epoch',start))
    # Validation is a held-out subset of seen-city/source-task training data.
    # It is NOT any of the SCUI/UCSI/UCUI test splits.
    if not 0<args.val_fraction<0.5:raise ValueError('--val-fraction must be (0, 0.5)')
    if len(ds)<3:raise ValueError('Need at least 3 samples to split train/validation')
    # Split by city + tile ID, not by CSV row: the same tile can have many task labels.
    if 'image_name' not in ds.df.columns:
        raise ValueError('image_name required for group-aware validation')
    group_col=(ds.df['city'].astype(str)+'::'+ds.df['image_name'].astype(str))
    unique_groups=np.array(sorted(set(group_col)))
    if len(unique_groups)<2:raise ValueError('Need >=2 distinct city/tile groups for validation')
    rng=np.random.default_rng(args.seed)
    rng.shuffle(unique_groups)
    n_val=max(1,int(len(unique_groups)*args.val_fraction))
    n_val=min(n_val,len(unique_groups)-1)
    validation_groups=set(unique_groups[:n_val])
    val_indices=[i for i,g in enumerate(group_col) if g in validation_groups]
    train_indices=[i for i,g in enumerate(group_col) if g not in validation_groups]
    tr=Subset(ds,train_indices);val=Subset(ds,val_indices)
    if args.resume and start>=args.epochs:
        print(f'Already reached epoch {start}; no further training required')
        return
    for ep in range(start,args.epochs):
        train_stats=loop(model,tr,args,provider,source_tasks,device,cache,optimizer)
        val_stats=loop(model,val,args,provider,source_tasks,device,cache,split_name='validation')
        val_mse=val_stats['overall']['mse']
        is_best=val_mse is not None and val_mse<best
        if is_best:
            best=val_mse
            best_epoch=ep+1
        metadata['best_epoch']=best_epoch
        save_checkpoint(out/'last.pt',model=model,optimizer=optimizer,cfg=cfg,meta=metadata,epoch=ep+1,best_val=best)
        if is_best:
            save_checkpoint(out/'best.pt',model=model,optimizer=optimizer,cfg=cfg,meta=metadata,epoch=ep+1,best_val=best)
        record={'epoch':ep+1,'train':train_stats,'validation':val_stats,'best_val_mse':best}
        with (out/'history.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
        print(epoch_text(record),flush=True)
        if getattr(args,'patience',0)>0 and (ep+1-best_epoch)>=args.patience:
            print(f'[early-stop] validation MSE has not improved for {args.patience} epochs; best epoch={best_epoch}',flush=True)
            break
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))

def evaluate(args):
    if not args.checkpoint:raise ValueError('evaluate requires --checkpoint')
    seed_all(args.seed)
    # Checkpoint governs architecture, source indicator ordering, and provider.
    raw=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    meta=raw['meta'];original=raw['config']
    args.provider=meta['provider'];args.model_id=meta['model_id']
    args.task_dim=original['task_dim'];args.visual_dim=original['visual_dim']
    args.hidden_dim=original['hidden_dim'];args.experts=original['n_experts'];args.top_k=original['top_k']
    source_tasks=tuple(meta['source_tasks'])
    print(f'[eval] loading provider={args.provider} model={args.model_id}',flush=True)
    cfg,provider,model,device,cache=make_components(args,source_tasks)
    load_checkpoint(args.checkpoint,model=model,device=device)
    print('[eval] checkpoint loaded',flush=True)
    names={'SCUI':args.scui_csv,'UCSI':args.ucsi_csv,'UCUI':args.ucui_csv}
    datasets={key:data_from_path(value,args,'eval') for key,value in names.items() if value}
    if not datasets:raise ValueError('Provide at least one test split CSV')
    # Compare metadata of seen-city/source-task training with test partitions.
    if args.train_csv:
        train_df=data_from_path(args.train_csv,args,'train').df
        check_splits(train_df,{name:ds.df for name,ds in datasets.items()})
    results={name:loop(model,ds,args,provider,source_tasks,device,cache,split_name=name) for name,ds in datasets.items()}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(results,indent=2))
    for name,result in results.items():print(f'{name} {metric_text(result[\'overall\'])}',flush=True)
    return results
