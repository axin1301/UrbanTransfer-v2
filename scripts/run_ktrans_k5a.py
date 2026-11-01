"""K5-A: nested task-heldout supervision for a low-capacity spatial router.

No pseudo-target labels enter the inner expert optimizer or checkpoint selection.
Fits only a 4-feature ridge router on *inner* pseudo-target transfer scores.
The *outer* heldout task is never used to fit the router or source experts.
Evaluation is development-only; outer last.pt avoids meta-label checkpoint selection.
"""
import argparse,csv,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from scripts.diagnose_k4_profiles import load_model
from urban_transfer.ktrans_text import qwen_task_vectors

ROOT=Path(__file__).resolve().parents[1]

def safe_corr(a,b):
    if len(a)<3 or np.std(a)<1e-8 or np.std(b)<1e-8:return 0.
    return float(np.corrcoef(a,b)[0,1])

def features(a,b,source_stats):
    sim=float(F.cosine_similarity(a.float()[None],b.float()[None]).item())
    mean,std=source_stats
    return np.array([1.,sim,mean/5.,np.log1p(std)],dtype=np.float64)

def fit_ridge(x,y,penalty=1.):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=4 or len(x)!=len(y):raise ValueError('Invalid training pairs')
    regularizer=np.diag([0.,penalty,penalty,penalty])
    return np.linalg.solve(x.T@x+regularizer,x.T@y)

def softmax(logits,temp):
    a=np.asarray(logits,dtype=float)/temp
    a=np.exp(a-a.max());return a/a.sum()

def extract_outputs(model,ds,indices,device,batch_size):
    cols=[[] for _ in model.source_tasks]
    targets=[]
    with torch.inference_mode():
        for batch in DataLoader(Subset(ds,indices),batch_size=batch_size,shuffle=False,num_workers=0):
            x=torch.cat((batch['image'].to(device).float(),
                         batch['context'].to(device).float()),dim=-1)
            h=model.shared(x)
            z=torch.cat([expert(h) for expert in model.experts],dim=-1).cpu().numpy()
            for i in range(z.shape[1]):cols[i].extend(z[:,i])
            targets.extend(batch['target'].numpy())
    return np.asarray(cols,dtype=float),np.asarray(targets,dtype=float)

def outer_predictions(model,ds,splits,device,batch_size,route,scale_route):
    # Evaluate a supplied spatial route directly, without mutating model buffers.
    # Scale always uses the original, frozen semantic route in this prototype.
    stats={}
    for protocol in ('meta_SCUI','meta_UCUI'):
        z,y=extract_outputs(model,ds,splits[protocol],device,batch_size)
        spatial=route@z
        if scale_route!='semantic':raise ValueError('K5-A freezes semantic scale routing')
        q=model._k5_original_route
        mean=float(q@model.source_means.cpu().numpy())
        std=float(q@model.source_stds.cpu().numpy())
        pred=mean+std*spatial
        var=float(np.var(y));mse=float(np.mean((pred-y)**2))
        stats[protocol]={'n':len(y),'mse':mse,'r2':1.-mse/var if var>1e-12 else None,
            'pearson':safe_corr(pred,y),'bias':float(np.mean(pred-y)),
            'pred_std':float(pred.std()),'target_std':float(y.std())}
    return stats

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outer-checkpoint',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--eval-batch-size',type=int,default=1024)
    p.add_argument('--ridge-penalty',type=float,default=1.)
    p.add_argument('--route-temperature',type=float,default=.15)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.ridge_penalty<=0 or args.route_temperature<=0:
        p.error('Invalid positive hyperparameter')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    meta,ds,splits,outer,device=load_model(args.outer_checkpoint,args.feature_dir,
        args.text_cache_dir,args.image_root)
    if meta.get('task_profile')!='structured':
        print('[k5a] warning: expected Qwen structured profile; got',meta.get('task_profile'))
    outer_target=meta['heldout_task']
    inner_tasks=list(outer.source_tasks)
    # Critical: nested CSV is strictly restricted to OUTER TRAIN rows:
    # no outer heldout task and no outer heldout cities.
    train_df=ds.df.iloc[splits['train']].copy()
    if outer_target in set(train_df.indicator_column):raise AssertionError('Outer label leaked')
    if set(train_df.indicator_column)!=set(inner_tasks):raise AssertionError('Unexpected training tasks')
    nested_csv=out/'inner_source_only.csv'
    train_df.to_csv(nested_csv,index=False)
    vectors=qwen_task_vectors(root=args.text_cache_dir,model_id=meta['text_model_id'],
                              profile=meta['task_profile'],only_cached=True)
    scored=[]
    for pseudo in inner_tasks:
        fold=out/'inner_models'/pseudo
        final=fold/'last.pt'
        if not (fold/'meta_protocol.json').is_file():
            cmd=[sys.executable,'-u','run_ktrans.py','train',
                '--train-csv',str(nested_csv),'--image-root',args.image_root,
                '--feature-dir',args.feature_dir,'--text-cache-dir',args.text_cache_dir,
                '--expert-bank','--normalized-experts','--task-encoder','qwen3',
                '--task-profile',meta['task_profile'],'--text-model-id',meta['text_model_id'],
                '--heldout-task',pseudo,'--limit',str(len(train_df)),
                '--meta-city-fraction','0.2','--seed',str(meta['seed']),
                '--batch-size',str(args.batch_size),'--eval-batch-size',str(args.eval_batch_size),
                '--epochs',str(args.epochs),'--patience','0',
                '--output',str(fold)]
            print('[k5a] train nested pseudo-target',pseudo,flush=True)
            with (out/f'{pseudo}.train.log').open('a') as log:
                subprocess.run(cmd,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
        if not final.is_file():raise FileNotFoundError(final)
        inner_meta,inner_ds,inner_splits,inner_model,inner_device=load_model(final,
            args.feature_dir,args.text_cache_dir,args.image_root)
        if pseudo!=inner_meta['heldout_task'] or outer_target in inner_model.source_tasks:
            raise AssertionError('Nested model source isolation broken')
        z,y=extract_outputs(inner_model,inner_ds,inner_splits['meta_SCUI'],
                             inner_device,args.eval_batch_size)
        for j,source in enumerate(inner_model.source_tasks):
            # For a *standardized* expert prediction, Pearson is invariant to
            # positive affine scaling, so this is spatial-only supervision.
            quality=safe_corr(z[j],y)
            x=features(vectors[pseudo],vectors[source],
                       inner_meta['source_stats'][source])
            scored.append({'pseudo_target':pseudo,'source_expert':source,
                           'spatial_pearson':quality,'features':x.tolist()})
            print(f'[k5a] {pseudo} <- {source}: Pearson={quality:.3f}',flush=True)
    X=np.asarray([r['features'] for r in scored]);Y=np.asarray([r['spatial_pearson'] for r in scored])
    beta=fit_ridge(X,Y,args.ridge_penalty)
    logits=np.array([features(vectors[outer_target],vectors[source],
                               meta['source_stats'][source])@beta
                     for source in inner_tasks])
    learned=softmax(logits,args.route_temperature)
    original=outer.routing[outer.task_to_all[outer_target]].detach().cpu().numpy().copy()
    outer._k5_original_route=original
    uniform=np.ones(len(inner_tasks))/len(inner_tasks)
    eval_rows=[]
    for route_name,route in [('semantic',original),('uniform',uniform),('k5a_ridge',learned)]:
        for protocol,v in outer_predictions(outer,ds,splits,device,args.eval_batch_size,
                                           route,'semantic').items():
            eval_rows.append({'route':route_name,'protocol':protocol,**v})
    with (out/'pseudo_transfer_scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['pseudo_target','source_expert','spatial_pearson'])
        w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in scored)
    with (out/'evaluation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(eval_rows[0]))
        w.writeheader();w.writerows(eval_rows)
    info={'outer_task':outer_target,'outer_checkpoint':str(args.outer_checkpoint),
          'inner_model_checkpoint':'last.pt (not target-selected best.pt)',
          'inner_source_tasks':inner_tasks,'ridge_weights':beta.tolist(),
          'route_temperature':args.route_temperature,'ridge_penalty':args.ridge_penalty,
          'semantic_weights':original.tolist(),'learned_weights':learned.tolist(),
          'warning':'Outer checkpoint best.pt was selected on meta evaluation labels; development-only'}
    (out/'protocol.json').write_text(json.dumps(info,indent=2))
    print('\nK5-A learned route vs frozen routes:')
    for source,a,b in zip(inner_tasks,original,learned):
        print(f'  {source:<30} semantic={a:.3f} k5a={b:.3f}')
    for row in eval_rows:
        print(f"  {row['route']:<12} {row['protocol']:<10} MSE={row['mse']:.3f} "
              f"R2={row['r2']:.3f} Pearson={row['pearson']:.3f} Bias={row['bias']:+.3f}")
    print('Saved:',out/'evaluation.csv')

if __name__=='__main__':main()
