"""Paired K4 checkpoint diagnostic: route weights and aligned spatial predictions.

No optimizer, no fitting to evaluation labels, no external target test splits.
Run with two independently trained checkpoints for the SAME heldout task.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from urban_transfer.ktrans import dataset,meta_indices,make_model
from urban_transfer.ktrans_text import qwen_task_vectors
from urban_transfer.checkpoints import load_checkpoint

def load_model(path,feature_dir,text_cache_dir):
    raw=torch.load(path,map_location='cpu',weights_only=False)
    meta=raw['meta'];cfg=raw['config']
    if not meta.get('expert_bank') or not meta.get('normalized_experts'):
        raise ValueError('Requires K3-A/K4 normalized source expert checkpoint')
    if meta['task_encoder']!='qwen3':
        raise ValueError('This paired diagnostic expects cached Qwen3 text vectors')
    args=SimpleNamespace(task_encoder='qwen3',heldout_task=meta['heldout_task'],
        seed=meta['seed'],limit=meta['limit'],meta_city_fraction=meta['meta_city_fraction'],
        remoteclip_model=meta['remoteclip_model'],text_model_id=meta['text_model_id'],
        task_profile=meta['task_profile'],llm_profile_file=None,
        text_cache_dir=text_cache_dir,feature_dir=feature_dir,train_csv=meta['train_csv'],
        image_root=None,image_column=None,scale_head=False,expert_bank=True,
        normalized_experts=True,router_temperature=meta['router_temperature'],
        visual_dim=cfg['visual_dim'],hidden_dim=cfg['hidden_dim'],
        experts=cfg['n_experts'],top_k=cfg['top_k'])
    vectors=qwen_task_vectors(root=text_cache_dir,model_id=args.text_model_id,
             profile=args.task_profile,only_cached=True)
    ds=dataset(args,vectors)
    splits,cities=meta_indices(ds.df,args.heldout_task,args.meta_city_fraction,args.seed)
    if cities!=meta['heldout_cities']:raise ValueError('Meta-city split differs from checkpoint')
    _,model,device=make_model(args,ds[0]['image'].numel(),cfg['task_dim'],
         cfg['n_sources'],vectors,meta['source_tasks'],meta['source_stats'])
    load_checkpoint(path,model=model,device=device)
    model.eval()
    return meta,ds,splits,model,device

def predictions(model,ds,splits,device,batch_size):
    result={}
    for protocol in ('meta_SCUI','meta_UCSI','meta_UCUI'):
        vals=[]
        loader=DataLoader(Subset(ds,splits[protocol]),batch_size=batch_size,
                          shuffle=False,num_workers=0)
        with torch.inference_mode():
            for batch in loader:
                x={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
                vals.extend(model(x).detach().cpu().numpy().ravel().tolist())
        result[protocol]=np.asarray(vals,dtype=float)
    return result

def corr(a,b):
    if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12:return float('nan')
    return float(np.corrcoef(a,b)[0,1])

def run(a_path,b_path,output,feature_dir,text_cache_dir,batch_size):
    a,da,sa,ma,device_a=load_model(a_path,feature_dir,text_cache_dir)
    b,db,sb,mb,device_b=load_model(b_path,feature_dir,text_cache_dir)
    checks=('heldout_task','seed','limit','meta_city_fraction','train_csv','heldout_cities','source_tasks')
    for key in checks:
        if a[key]!=b[key]:raise ValueError(f'Unpaired checkpoint metadata: {key}')
    if len(da.df)!=len(db.df) or not da.df[['city','image_name','indicator_column']].equals(
            db.df[['city','image_name','indicator_column']]):
        raise ValueError('Sample order differs; cannot compute paired metrics')
    pa=predictions(ma,da,sa,device_a,batch_size)
    pb=predictions(mb,db,sb,device_b,batch_size)
    name=a['heldout_task']
    rows=[]
    for label,model in [('short',ma),('structured',mb)]:
        idx=model.task_to_all[name]
        weights=model.routing[idx].detach().cpu().tolist()
        for expert,w,mean,std in zip(model.source_tasks,weights,
                                     model.source_means.cpu().tolist(),
                                     model.source_stds.cpu().tolist()):
            rows.append({'heldout_task':name,'profile':label,'source_expert':expert,
                         'semantic_route_weight':w,'source_train_mean':mean,
                         'source_train_std':std})
    output.mkdir(parents=True,exist_ok=True)
    with (output/'expert_routes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    results=[]
    for protocol in pa:
        x,y=pa[protocol],pb[protocol]
        if len(x)!=len(y):raise ValueError('Prediction lengths differ')
        c=corr(x,y)
        slope=float(np.cov(x,y,bias=True)[0,1]/np.var(x)) if np.var(x)>1e-12 else float('nan')
        intercept=float(y.mean()-slope*x.mean())
        residual=y-(slope*x+intercept)
        results.append({'heldout_task':name,'protocol':protocol,'n':len(x),
          'pred_corr_short_structured':c,'short_mean':float(x.mean()),
          'structured_mean':float(y.mean()),'mean_shift':float(y.mean()-x.mean()),
          'short_std':float(x.std()),'structured_std':float(y.std()),
          'std_ratio':float(y.std()/x.std()) if x.std()>1e-12 else float('nan'),
          'affine_slope':slope,'affine_intercept':intercept,
          'affine_residual_std':float(residual.std())})
    with (output/'paired_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    print(f'\nTask: {name}\nSemantic expert routes:')
    for r in rows:
        print(f"  {r['profile']:<11} {r['source_expert']:<31} {r['semantic_route_weight']:.3f}")
    print('\nPaired prediction diagnostic:')
    for r in results:
        print(f"  {r['protocol']}: corr={r['pred_corr_short_structured']:.3f} "
              f"mean_shift={r['mean_shift']:+.3f} std_ratio={r['std_ratio']:.3f} "
              f"affine_residual_std={r['affine_residual_std']:.3f}")
    print('Saved:',output/'expert_routes.csv','and',output/'paired_predictions.csv')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--short-checkpoint',required=True)
    p.add_argument('--structured-checkpoint',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--batch-size',type=int,default=1024)
    args=p.parse_args()
    run(args.short_checkpoint,args.structured_checkpoint,args.output,
        args.feature_dir,args.text_cache_dir,args.batch_size)
if __name__=='__main__':main()
