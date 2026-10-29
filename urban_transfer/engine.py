"""End-to-end train/evaluate; online MLLM called for every batch item (no hidden persistent caching)."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import ModelConfig, TransferModel
from .vision import SmallVisionEncoder
from .knowledge import MockProvider
from .qwen3vl import Qwen3VLProvider
from .data_m1 import ImageUrbanDataset

class Predictor(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__();self.vision=SmallVisionEncoder(cfg.visual_dim);self.transfer=TransferModel(cfg)
    def forward(self,b,knowledge):
        visual=self.vision(b['image']);return self.transfer(visual,b['context'],b['task'],knowledge)

def batch_knowledge(batch,provider,source_tasks,device):
    data=[]
    for city,path,desc in zip(batch['city'],batch['image_path'],batch['task_description']):
        k=provider.infer(image_path=path,city=city,task_description=desc,source_tasks=source_tasks)
        data.append(k.relevance+k.morphology+[k.uncertainty])
    return torch.tensor(data,dtype=torch.float32,device=device)

def metrics(y,p):
    a=np.array(y,dtype=float);b=np.array(p,dtype=float)
    if not len(a):return {'n':0,'mse':None,'mae':None,'r2':None}
    mse=float(np.mean((a-b)**2));mae=float(np.mean(np.abs(a-b)))
    den=float(((a-a.mean())**2).sum());r2=float(1-((a-b)**2).sum()/den) if den>0 else None
    return {'n':len(a),'mse':mse,'mae':mae,'r2':r2}

def build_provider(mode,model_id):
    if mode=='mock': return MockProvider()
    if mode=='qwen3vl': return Qwen3VLProvider(model_id=model_id)
    raise ValueError(mode)

def run(args):
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.provider=='qwen3vl' and device.type!='cuda': raise RuntimeError('Qwen3-VL mode requires ROCm GPU.')
    if args.provider=='qwen3vl' and args.allow_missing_images: raise ValueError('Cannot use Qwen3-VL with missing images')
    paths={'train':args.train_csv,'SCUI':args.scui_csv,'UCSI':args.ucsi_csv,'UCUI':args.ucui_csv}
    datasets={name:ImageUrbanDataset(path,args.image_root,partition='train' if name=='train' else 'eval',
                limit=args.limit,require_images=not args.allow_missing_images,image_column=args.image_column) for name,path in paths.items() if path}
    train_cities=set(datasets['train'].df.city)
    for protocol in ('UCSI','UCUI'):
        if protocol in datasets and train_cities.intersection(set(datasets[protocol].df.city)):
            raise ValueError(f'City leakage in {protocol}')
    train_tasks=set(datasets['train'].df.indicator_column)
    for protocol in ('SCUI','UCUI'):
        if protocol in datasets and train_tasks.intersection(set(datasets[protocol].df.indicator_column)):
            raise ValueError(f'Task leakage in {protocol}')
    source_tasks=tuple(sorted(train_tasks))
    cfg=ModelConfig(use_knowledge=True,n_sources=len(source_tasks),task_dim=args.task_dim,visual_dim=args.visual_dim,
           hidden_dim=args.hidden_dim,n_experts=args.experts,top_k=args.top_k)
    predictor=Predictor(cfg).to(device)
    optimizer=torch.optim.AdamW(predictor.parameters(),lr=args.lr,weight_decay=1e-4)
    provider=build_provider(args.provider,args.model_id)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    meta={'source_tasks':source_tasks,'provider':args.provider,'model_id':args.model_id,'source_data':paths,'seed':args.seed,
          'task_vectors':'deterministic hash bag-of-words baseline (not pretrained semantics)', 'image_root':args.image_root, 'image_column':args.image_column}
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    for epoch in range(args.epochs):
        predictor.train(); losses=[]
        loader=DataLoader(datasets['train'],batch_size=args.batch_size,shuffle=True,num_workers=0)
        for batch in loader:
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            knowledge=batch_knowledge(batch,provider,source_tasks,device)
            y=predictor(batch,knowledge);loss=torch.nn.functional.mse_loss(y,batch['target'])
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(loss.item())
        print(f'epoch={epoch+1} mean_mse_loss={np.mean(losses):.5f}',flush=True)
        torch.save({'model':predictor.state_dict(),'config':cfg.__dict__,'meta':meta,'epoch':epoch+1},out/'last.pt')
    predictor.eval(); results={}
    for protocol,ds in datasets.items():
        if protocol=='train':continue
        ys=[];ps=[];by_task={}
        for batch in DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0):
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            knowledge=batch_knowledge(batch,provider,source_tasks,device)
            with torch.no_grad(): p=predictor(batch,knowledge)
            for target,pred,task in zip(batch['target'].tolist(),p.detach().cpu().tolist(),batch['task_name']):
                ys.append(target);ps.append(pred)
                if task not in by_task:by_task[task]=([],[])
                by_task[task][0].append(target);by_task[task][1].append(pred)
        results[protocol]={'overall':metrics(ys,ps),'per_task':{t:metrics(*v) for t,v in by_task.items()}}
        print(protocol,results[protocol]['overall'],flush=True)
    (out/'metrics.json').write_text(json.dumps(results,indent=2))
    return results
