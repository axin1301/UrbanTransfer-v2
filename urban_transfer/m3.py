"""M3 training/evaluation. M2 checkpoints, code and metrics remain untouched."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from .m2 import seed_all,loop,source_info,check_splits
from .model import ModelConfig,TransferModel
from .engine import build_provider
from .cache import HiddenStateCache
from .checkpoints import save_checkpoint,load_checkpoint
from .formatting import epoch_text,metric_text
from .m3_features import FeatureStore,CachedFeatureDataset,precompute

class M3Predictor(nn.Module):
    def __init__(self,cfg,input_dim,latent_dim=None):
        super().__init__()
        self.projector=nn.Sequential(nn.LayerNorm(input_dim),
                                     nn.Linear(input_dim,cfg.visual_dim),nn.GELU())
        self.transfer=TransferModel(cfg)
        self.latent_projection=(nn.Sequential(nn.LayerNorm(latent_dim),
                        nn.Linear(latent_dim,cfg.n_sources+4)) if latent_dim else None)
    def forward(self,batch,knowledge):
        visual=self.projector(batch['image'].float())
        if self.latent_projection is not None:
            knowledge=torch.sigmoid(self.latent_projection(knowledge.float()))
        return self.transfer(visual,batch['context'],batch['task'],knowledge)

def store_for(args):
    return FeatureStore(args.feature_dir,encoder_id='RemoteCLIP-'+args.remoteclip_model)

def dataset(args,path,store,partition='eval'):
    return CachedFeatureDataset(path,args.image_root,store,partition=partition,
        limit=args.limit,task_dim=args.task_dim,image_column=args.image_column)

def components(args,source_tasks,feature_dim):
    cfg=ModelConfig(use_knowledge=(args.provider!='no-llm'),
         n_sources=len(source_tasks),task_dim=args.task_dim,visual_dim=args.visual_dim,
         hidden_dim=args.hidden_dim,n_experts=args.experts,top_k=args.top_k)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    provider=None if args.provider=='no-llm' else build_provider(
         'qwen3vl-latent' if args.provider=='shuffled-latent' else args.provider,args.model_id)
    latent_dim=(provider.model.config.text_config.hidden_size if provider else None)
    model=M3Predictor(cfg,feature_dim,latent_dim).to(device)
    cache=HiddenStateCache(args.cache_dir,args.model_id,
         enabled=bool(provider) and not args.no_cache)
    return cfg,provider,model,device,cache

def train(args):
    seed_all(args.seed)
    store=store_for(args)
    ds=dataset(args,args.train_csv,store,'train')
    source_tasks=source_info(ds.df)
    feature_dim=store.get(ds.paths[0]).numel()
    cfg,provider,model,device,cache=components(args,source_tasks,feature_dim)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if not 0<args.val_fraction<0.5:raise ValueError('--val-fraction must be in (0,0.5)')
    group=ds.df.city.astype(str)+'::'+ds.df.image_name.astype(str)
    unique=np.array(sorted(set(group)))
    if len(unique)<2:raise ValueError('Need at least two unique tiles')
    rng=np.random.default_rng(args.seed);rng.shuffle(unique)
    n_val=min(max(1,int(len(unique)*args.val_fraction)),len(unique)-1)
    val_groups=set(unique[:n_val])
    tr=Subset(ds,[i for i,g in enumerate(group) if g not in val_groups])
    val=Subset(ds,[i for i,g in enumerate(group) if g in val_groups])
    best=float('inf');best_epoch=0;start=0
    meta={'pipeline':'m3-frozen-remoteclip','provider':args.provider,'model_id':args.model_id,
          'remoteclip_model':args.remoteclip_model,'feature_dim':feature_dim,
          'source_tasks':list(source_tasks),'seed':args.seed,
          'train_csv':str(Path(args.train_csv).resolve()),'limit':args.limit,
          'val_fraction':args.val_fraction}
    if args.resume:
        ck=load_checkpoint(args.resume,model=model,optimizer=opt,restore_rng=True,device=device)
        for field in ('pipeline','provider','remoteclip_model','model_id','source_tasks','train_csv','seed','limit','val_fraction','feature_dim'):
            if ck['meta'].get(field)!=meta.get(field):raise ValueError('Resume mismatch: '+field)
        start=ck['epoch'];best=ck.get('best_val',float('inf'))
        best_epoch=int(ck['meta'].get('best_epoch',start))
    for epoch in range(start,args.epochs):
        training=loop(model,tr,args,provider,source_tasks,device,cache,optimizer=opt)
        validation=loop(model,val,args,provider,source_tasks,device,cache,split_name='validation')
        score=validation['overall']['mse']
        improved=score is not None and score<best
        if improved:best=score;best_epoch=epoch+1
        meta['best_epoch']=best_epoch
        save_checkpoint(out/'last.pt',model=model,optimizer=opt,cfg=cfg,meta=meta,epoch=epoch+1,best_val=best)
        if improved:save_checkpoint(out/'best.pt',model=model,optimizer=opt,cfg=cfg,meta=meta,epoch=epoch+1,best_val=best)
        record={'epoch':epoch+1,'train':training,'validation':validation,'best_val_mse':best}
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(epoch_text(record),flush=True)
        if args.patience>0 and epoch+1-best_epoch>=args.patience:break

def evaluate(args):
    if not args.checkpoint:raise ValueError('--checkpoint required')
    seed_all(args.seed)
    raw=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    meta=raw['meta']
    if meta.get('pipeline')!='m3-frozen-remoteclip':raise ValueError('This is not an M3 checkpoint')
    args.provider=meta['provider'];args.model_id=meta['model_id']
    if args.remoteclip_model!=meta['remoteclip_model']:
        raise ValueError('RemoteCLIP model differs from checkpoint')
    cfg_data=raw['config']
    for k,arg in [('task_dim','task_dim'),('visual_dim','visual_dim'),('hidden_dim','hidden_dim'),
                  ('n_experts','experts'),('top_k','top_k')]:
        setattr(args,arg,cfg_data[k])
    store=store_for(args)
    source_tasks=tuple(meta['source_tasks'])
    cfg,provider,model,device,cache=components(args,source_tasks,meta['feature_dim'])
    load_checkpoint(args.checkpoint,model=model,device=device)
    names={'SCUI':args.scui_csv,'UCSI':args.ucsi_csv,'UCUI':args.ucui_csv}
    ds={k:dataset(args,v,store) for k,v in names.items() if v}
    if not ds:raise ValueError('Specify at least one test CSV')
    if args.train_csv:
        train_df=dataset(args,args.train_csv,store,'train').df
        check_splits(train_df,{k:d.df for k,d in ds.items()})
    results={}
    for name,d in ds.items():
        results[name]=loop(model,d,args,provider,source_tasks,device,cache,split_name=name)
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        (out/'metrics.json').write_text(json.dumps(results,indent=2))
        print(f'{name} {metric_text(results[name][\'overall\'])}',flush=True)
    return results

def extract(args):
    paths=[p for p in (args.train_csv,args.scui_csv,args.ucsi_csv,args.ucui_csv) if p]
    if not paths:raise ValueError('Supply CSV(s) for precompute')
    precompute(paths,args.image_root,store_for(args),model_name=args.remoteclip_model,
        checkpoint=args.remoteclip_checkpoint,batch_size=args.batch_size,
        image_column=args.image_column,limit=args.limit)
