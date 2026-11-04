"""K7-D stage 3: source-only four-way multimodal expert-route ablation.

Freeze K4 normalized experts and K5-A route; fit only lightweight
knowledge-conditioned source-expert mixtures. Own-task experts are
excluded on outer source train samples. No outer heldout task labels
enter training/checkpoint selection, but frozen outer shared features
ARE NOT independently inner cross-fitted. Report DEV-only metrics.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from scripts.diagnose_k4_profiles import load_model
from scripts.run_ktrans_k6 import metrics
from scripts.cache_ktrans_k7d import merge_records,unique_jobs
from scripts.run_ktrans_k7b_controls import SYSTEM,TEXT_ONLY_SYSTEM,build_prompt
from urban_transfer.ktrans_k7d import AnchoredKnowledgeRouter,knowledge_vector,VARIANTS
from urban_transfer.ktrans_k61 import build_source_transfer_routes
from urban_transfer.ktrans_text import qwen_task_vectors
from urban_transfer.task_profiles import task_texts

SPLITS=('train','meta_SCUI','meta_UCSI','meta_UCUI')

def validate_manifest(manifest,ds,splits,meta,k4):
    if manifest.get('version')!='k7d-manifest-v1':raise ValueError('Unknown K7-D manifest')
    if Path(manifest['k4_checkpoint']).resolve()!=Path(k4).resolve():
        raise ValueError('Manifest K4 checkpoint mismatch')
    if manifest['heldout_task']!=meta['heldout_task']:
        raise ValueError('Manifest target task mismatch')
    if tuple(manifest['source_tasks'])!=tuple(meta['source_tasks']):
        raise ValueError('Manifest source task order mismatch')
    allowed={p:set(splits[p]) for p in SPLITS}
    seen=set()
    for row in manifest['rows']:
        ix=int(row['index']);p=row['partition']
        if p not in allowed or ix not in allowed[p]:
            raise ValueError('Manifest row outside checkpoint protocol partition')
        if (ix,p) in seen:raise ValueError('Duplicate manifest row')
        seen.add((ix,p))
        frame=ds.df.iloc[ix]
        if str(frame.indicator_column)!=row['task'] or str(frame.city)!=row['city']:
            raise ValueError('Manifest sample identity changed')
        if str(Path(ds.base.paths[ix]).resolve())!=row['image_path']:
            raise ValueError('Manifest image path differs from RemoteCLIP input')
    if not {r['partition'] for r in manifest['rows']}==set(SPLITS):
        raise ValueError('Manifest missing a split')
    if any(row['task']==meta['heldout_task'] for row in manifest['rows'] if row['partition']=='train'):
        raise AssertionError('Heldout task in source train set')

def index_knowledge(jobs,records):
    result={}
    for job in jobs:
        row=records.get(job['key'])
        if row is None or row.get('parsed') is None:
            raise ValueError('K7-D knowledge cache incomplete; run cache script. '
                             'Example missing '+job['condition']+' task='+job['task']+
                             ' tile='+str(job['tile_id']))
        if row.get('condition')!=job['condition'] or row.get('task')!=job['task'] or row.get('prompt')!=job['prompt']:
            raise ValueError('Cached VLM evidence provenance does not match manifest')
        result[(job['tile_id'],job['task'],job['condition'])]=row['parsed']
    return result

def vectors_for(rows,knowledge,variant):
    features=[]
    for row in rows:
        tile,task=row['tile_id'],row['task']
        feature=knowledge_vector(
            knowledge[(tile,task,'image_task')],
            knowledge[(tile,'image_only','image_only')],
            knowledge[(None,task,'text_only')],variant)
        features.append(feature)
    return torch.stack(features)

def extract_expert_batches(ds,rows,bank,device,batch_size):
    selected=[int(r['index']) for r in rows]
    outputs=[];targets=[]
    bank.eval()
    with torch.inference_mode():
        for batch in DataLoader(Subset(ds,selected),batch_size=batch_size,
                                shuffle=False,num_workers=0):
            x=torch.cat([batch['image'].to(device).float(),
                         batch['context'].to(device).float()],dim=-1)
            h=bank.shared(x)
            z=torch.cat([expert(h) for expert in bank.experts],dim=-1)
            outputs.append(z.detach().cpu())
            targets.append(batch['target'].detach().cpu())
    return torch.cat(outputs),torch.cat(targets)

def train_variant(variant,data,source_routes,target_route,bank,epochs,lr,seed,penalty,batch_size,device):
    torch.manual_seed(seed)
    n=len(bank.source_tasks)
    net=AnchoredKnowledgeRouter(n).to(device)
    # For comparability, hold model width, optimizer, epoch count and
    # seed fixed across all three knowledge ablations.
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
    subset=data['train']
    features=subset['features'][variant].to(device)
    expert=subset['experts'].to(device)
    labels=subset['labels'].to(device)
    ids=torch.tensor([bank.task_to_source[r['task']] for r in subset['rows']],device=device)
    prior=source_routes.to(device)[ids]
    own=torch.nn.functional.one_hot(ids,n).bool()
    mean=bank.source_means.to(device)[ids]
    std=bank.source_stds.to(device)[ids]
    yt=(labels-mean)/std
    if len(ids)<n*4:
        raise ValueError('Too few training rows for source-task routing')
    for epoch in range(1,epochs+1):
        net.train();loss_total=0.;count=0
        order=torch.randperm(len(ids),generator=torch.Generator().manual_seed(seed+epoch))
        for ix in order.split(batch_size):
            ix=ix.to(device)
            weights,gate=net(features[ix],prior[ix],own[ix])
            spatial=(weights*expert[ix]).sum(-1)
            loss=(spatial-yt[ix]).square().mean()+penalty*gate.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total+=float(loss.detach())*len(ix);count+=len(ix)
        print(f'[k7d] {variant:<11} epoch={epoch:02d} '
              f'source_only_loss={loss_total/count:.3f}',flush=True)
    return net

def evaluate(net,data,bank,source_routes,target_route,device,batch_size):
    metrics_rows=[]
    with torch.inference_mode():
        for name in ('meta_SCUI','meta_UCSI','meta_UCUI'):
            split=data[name];rows=split['rows']
            features=split['current_features'].to(device)
            expert=split['experts'].to(device)
            labels=split['labels'].numpy()
            n=len(bank.source_tasks)
            own=[bank.task_to_source.get(r['task']) for r in rows]
            target=torch.as_tensor(target_route,dtype=torch.float32,device=device)
            src=source_routes.to(device)
            routes=torch.stack([target if i is None else src[i] for i in own])
            masked=torch.zeros(len(own),n,dtype=torch.bool,device=device)
            for j,i in enumerate(own):
                if i is not None:masked[j,i]=True
            semantic=torch.stack([
                bank.routing[bank.task_to_all[r['task']]].to(device) for r in rows])
            means=semantic@bank.source_means.to(device)
            stds=semantic@bank.source_stds.to(device)
            # Source-only UCSI uses own-task moments, as in K6.1 evaluation.
            for j,i in enumerate(own):
                if i is not None:
                    means[j]=bank.source_means[i]
                    stds[j]=bank.source_stds[i]
            pred=[];gates=[]
            for ii in torch.arange(len(rows),device=device).split(batch_size):
                if net is None:
                    w=routes[ii]
                    g=torch.zeros(len(ii),device=device)
                else:
                    w,g=net(features[ii],routes[ii],masked[ii])
                p=means[ii]+stds[ii]*(expert[ii]*w).sum(-1)
                pred.extend(p.cpu().numpy().tolist())
                gates.extend(g.cpu().numpy().tolist())
            stats=metrics(labels,np.asarray(pred))
            metrics_rows.append({'protocol':name,**stats,
                                 'gate_mean':float(np.mean(gates)),
                                 'gate_std':float(np.std(gates))})
    return metrics_rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--k5a-protocol',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--knowledge-cache',type=Path,required=True)
    p.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--eval-batch-size',type=int,default=512)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--gate-penalty',type=float,default=.01)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.eval_batch_size<1 or args.lr<=0 or args.gate_penalty<0:
        p.error('Invalid training hyperparameters')
    meta,ds,splits,bank,device=load_model(args.checkpoint,args.feature_dir,
                                         args.text_cache_dir,args.image_root)
    manifest=json.loads(args.manifest.read_text())
    validate_manifest(manifest,ds,splits,meta,args.checkpoint)
    info=json.loads(args.k5a_protocol.read_text())
    if Path(info['outer_checkpoint']).resolve()!=args.checkpoint.resolve():
        raise ValueError('K5-A checkpoint mismatch')
    if info['outer_task']!=meta['heldout_task'] or tuple(info['inner_source_tasks'])!=bank.source_tasks:
        raise ValueError('K5-A outer task or expert ordering mismatch')
    vectors=qwen_task_vectors(root=args.text_cache_dir,model_id=meta['text_model_id'],
               profile=meta['task_profile'],only_cached=True)
    source_routes=build_source_transfer_routes(vectors,list(bank.source_tasks),
               meta['source_stats'],info['ridge_weights'],info['route_temperature'])
    target_route=torch.as_tensor(info['learned_weights'],dtype=torch.float32)
    if len(target_route)!=len(bank.source_tasks):raise ValueError('K5-A route dimensionality differs')
    jobs=unique_jobs(manifest,args.model,args.max_pixels)
    cache=merge_records([args.reuse_cache,args.knowledge_cache])
    knowledge=index_knowledge(jobs,cache)  # Explicit 100% coverage check
    grouped={part:[r for r in manifest['rows'] if r['partition']==part] for part in SPLITS}
    data={}
    for part,rows in grouped.items():
        z,y=extract_expert_batches(ds,rows,bank,device,args.eval_batch_size)
        data[part]={'rows':rows,'experts':z,'labels':y,
             'features':{v:vectors_for(rows,knowledge,v) for v in VARIANTS if v!='k5a'}}
        if part=='train' and set(r['task'] for r in rows)!=set(bank.source_tasks):
            raise AssertionError('Source-only task coverage missing')
    if any(r['task']==meta['heldout_task'] for r in grouped['train']):
        raise AssertionError('Outer heldout task leaked into K7-D router optimization')
    args.output.mkdir(parents=True,exist_ok=True)
    scores=[]
    for variant in VARIANTS:
        net=None if variant=='k5a' else train_variant(variant,data,source_routes,
            target_route,bank,args.epochs,args.lr,args.seed,args.gate_penalty,
            args.batch_size,device)
        for part in ('meta_SCUI','meta_UCSI','meta_UCUI'):
            # Populate the per-variant input, kept separate from others.
            data[part]['current_features']=(torch.zeros(len(data[part]['rows']),36)
                  if variant=='k5a' else data[part]['features'][variant])
        for row in evaluate(net,data,bank,source_routes,target_route,device,args.eval_batch_size):
            result={'variant':variant,'outer_task':meta['heldout_task'],**row}
            scores.append(result)
            print(f'[k7d] {variant:<11} {row["protocol"]:<10} '
                  f'MSE={row["mse"]:.3f} Pearson={row["pearson"]:.3f} '
                  f'Bias={row["bias"]:+.3f} gate_std={row["gate_std"]:.3f}',flush=True)
        if net is not None:
            torch.save({'variant':variant,'router':{k:v.cpu() for k,v in net.state_dict().items()},
                'k4_checkpoint':str(args.checkpoint),'k5a_protocol':str(args.k5a_protocol),
                'manifest':str(args.manifest),'seed':args.seed,'epochs':args.epochs},
                args.output/(variant+'_last.pt'))
    with (args.output/'evaluation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(scores[0]))
        w.writeheader();w.writerows(scores)
    (args.output/'protocol.json').write_text(json.dumps({
        'version':'k7d-source-only-dev-v1','outer_task':meta['heldout_task'],
        'k4_checkpoint':str(args.checkpoint.resolve()),
        'k5a_protocol':str(args.k5a_protocol.resolve()),
        'manifest':str(args.manifest.resolve()),
        'partitions':manifest['partition_counts'],'epochs':args.epochs,'seed':args.seed,
        'source_only_training':True,'own_task_expert_masked':True,
        'heldout_task_optimized':False,'model_selection':'fixed last epoch only',
        'variants':list(VARIANTS),
        'warning':'DEV ONLY: outer K4 best.pt was meta-selected and the shared K4 trunk is NOT strictly inner crossfitted. Sampled meta rows must not guide K7 model choices.'
    },indent=2))
    print('[k7d] saved',args.output/'evaluation.csv',flush=True)

if __name__=='__main__':main()
