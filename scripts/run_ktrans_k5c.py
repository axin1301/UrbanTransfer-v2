"""Evaluate independent source-only K5-C scale calibration on frozen K5-B2."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from scripts.diagnose_k4_profiles import load_model
from scripts.run_ktrans_k5a import safe_corr
from urban_transfer.ktrans_k5b import RegionTaskComposer
from urban_transfer.ktrans_k5c import IndependentScaleRouter
from urban_transfer.ktrans_text import qwen_task_vectors

def metrics(y,p):
    mse=float(np.mean((y-p)**2));var=float(np.var(y))
    return {'n':len(y),'mse':mse,'r2':1-mse/var if var>1e-12 else float('nan'),
            'pearson':safe_corr(y,p),'bias':float(np.mean(p-y)),
            'pred_std':float(np.std(p))}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--composer-checkpoint',required=True,type=Path)
    p.add_argument('--output',default='runs/k5c_scale',type=Path)
    p.add_argument('--eval-batch-size',default=1024,type=int)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    saved=torch.load(args.composer_checkpoint,map_location='cpu',weights_only=False)
    meta,ds,splits,bank,device=load_model(saved['checkpoint'],args.feature_dir,
        args.text_cache_dir,args.image_root)
    vectors=qwen_task_vectors(root=args.text_cache_dir,model_id=meta['text_model_id'],
        profile=meta['task_profile'],only_cached=True)
    composer=RegionTaskComposer(bank,vectors,rank=saved['rank'],
                                 adapter_cap=saved.get('adapter_cap')).to(device)
    missing,unexpected=composer.load_state_dict(saved['composer'],strict=False)
    if any(not key.startswith('bank.') for key in missing) or unexpected:
        raise RuntimeError(f'Unexpected state keys: {missing} {unexpected}')
    composer.eval()
    target=meta['heldout_task']
    if target in meta['source_stats'] or target in bank.source_tasks:
        raise AssertionError('Outer target leaked into source statistics')
    cal=IndependentScaleRouter(vectors,meta['source_stats'])
    pred_mean,pred_std=cal.predict(target)
    # K4 and K5-B2 spatial routes share the same frozen semantic scale,
    # compare ONLY scale changes on identical K5-B2 spatial predictions.
    idx=bank.task_to_all[target]
    prior=bank.routing[idx].detach().cpu().numpy()
    mu=float(prior@bank.source_means.cpu().numpy())
    sigma=float(prior@bank.source_stds.cpu().numpy())
    rows=[]
    for protocol in ('meta_SCUI','meta_UCUI'):
        vals=[];spatial=[]
        with torch.inference_mode():
            for batch in DataLoader(Subset(ds,splits[protocol]),
                                    batch_size=args.eval_batch_size,num_workers=0):
                batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
                _,aux=composer(batch,return_aux=True)
                spatial.extend(aux['spatial'].cpu().numpy().tolist())
                vals.extend(batch['target'].cpu().numpy().tolist())
        y=np.asarray(vals);z=np.asarray(spatial)
        for name,m,s in [('k5b2_semantic_scale',mu,sigma),
                         ('k5c_learned_scale',pred_mean,pred_std)]:
            result={'variant':name,'protocol':protocol,**metrics(y,m+s*z),
                    'scale_mean':m,'scale_std':s}
            rows.append(result)
            print(f'{name:<22} {protocol:<10} MSE={result["mse"]:.3f} '
                  f'R2={result["r2"]:.3f} Pearson={result["pearson"]:.3f} '
                  f'Bias={result["bias"]:+.3f}',flush=True)
    with (args.output/'evaluation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/'protocol.json').write_text(json.dumps({
        'heldout_task':target,'source_tasks':list(cal.sources),
        'k5b2_checkpoint':str(args.composer_checkpoint),
        'frozen_semantic_moments':[mu,sigma],
        'learned_moments':[pred_mean,pred_std],
        'scale_ridge_coef':cal.coef.tolist(),'penalty':cal.penalty,
        'warning':'Development data for evaluation ONLY; outer K4 checkpoint was meta-selected'
    },indent=2))
    print(f'Scale moments: semantic=({mu:.3f},{sigma:.3f}) '
          f'learned=({pred_mean:.3f},{pred_std:.3f})')
    print('Saved',args.output/'evaluation.csv')

if __name__=='__main__':main()
