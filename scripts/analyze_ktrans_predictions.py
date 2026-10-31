"""K-TransMoE checkpoint prediction diagnostics on INTERNAL meta partitions only.

Per-partition true/predicted distribution, bias, Pearson, R2, baseline MSE.
Never uses external test splits or labels for calibration.
"""
import argparse
import sys
import csv
import json
from pathlib import Path

# Support direct execution: python scripts/analyze_ktrans_predictions.py
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from urban_transfer.data_paths import DEFAULT_SPLIT_DIR
from urban_transfer.data_m1 import task_vector
from urban_transfer.protocol import TASK_DESCRIPTIONS
from urban_transfer.ktrans import task_embeddings, dataset, meta_indices, make_model
from urban_transfer.checkpoints import load_checkpoint


def summaries(actual, predicted):
    y=np.asarray(actual,dtype=np.float64)
    p=np.asarray(predicted,dtype=np.float64)
    if len(y)==0:raise ValueError('Empty evaluation set')
    mse=float(np.mean((p-y)**2))
    mae=float(np.mean(np.abs(p-y)))
    target_mean=float(y.mean())
    pred_mean=float(p.mean())
    target_var=float(np.mean((y-target_mean)**2))
    pred_var=float(np.mean((p-pred_mean)**2))
    # This diagnostic oracle uses the target partition's own mean. DO NOT
    # use this number to calibrate outputs or call it a deployable baseline.
    oracle_mse=target_var
    # A fixed constant with the model's predicted mean, showing bias costs.
    pred_mean_mse=float(np.mean((y-pred_mean)**2))
    if target_var>1e-12 and pred_var>1e-12:
        pearson=float(np.corrcoef(y,p)[0,1])
    else:
        pearson=None
    return {
        'n':len(y),'target_mean':target_mean,'pred_mean':pred_mean,
        'target_std':float(np.std(y)),'pred_std':float(np.std(p)),
        'bias':float(np.mean(p-y)),'mse':mse,'mae':mae,
        'r2':float(1-mse/target_var) if target_var>1e-12 else None,
        'pearson':pearson,
        'oracle_mean_mse':oracle_mse,
        'pred_mean_constant_mse':pred_mean_mse,
        'spatial_gain_vs_pred_mean_constant':pred_mean_mse-mse,
    }


def evaluate_checkpoint(path,cli):
    raw=torch.load(path,map_location='cpu',weights_only=False)
    meta=raw['meta']
    if meta.get('pipeline')!='ktrans-k0-k1':
        raise ValueError(f'Unsupported checkpoint pipeline for {path}')
    args=SimpleNamespace(
        task_encoder=meta['task_encoder'],heldout_task=meta['heldout_task'],
        seed=meta['seed'],limit=meta['limit'],meta_city_fraction=meta['meta_city_fraction'],
        remoteclip_model=meta['remoteclip_model'],
        remoteclip_checkpoint=cli.remoteclip_checkpoint,
        text_cache_dir=cli.text_cache_dir,feature_dir=cli.feature_dir,
        train_csv=cli.train_csv or meta['train_csv'],
        image_root=cli.image_root,image_column=cli.image_column,
        scale_head=meta.get('scale_head',False),
        expert_bank=meta.get('expert_bank',False),
        normalized_experts=meta.get('normalized_experts',False),
        router_temperature=meta.get('router_temperature',0.15),
    )
    cfg=raw['config']
    for key in ('visual_dim','hidden_dim'):
        setattr(args,key,cfg[key])
    args.experts=cfg['n_experts']
    args.top_k=cfg['top_k']
    if args.task_encoder=='remoteclip':
        vectors=task_embeddings(args)
    else:
        vectors={task:torch.from_numpy(task_vector(desc,128).copy())
                 for task,desc in TASK_DESCRIPTIONS.items()}
    ds=dataset(args,vectors)
    splits,cities=meta_indices(ds.df,args.heldout_task,args.meta_city_fraction,args.seed)
    if cities!=meta['heldout_cities']:
        raise ValueError('Meta city split changed')
    expected_train_csv=Path(meta['train_csv']).resolve()
    if Path(args.train_csv).resolve()!=expected_train_csv:
        raise ValueError('Different training CSV; data split would not match checkpoint')
    config_task_dim=cfg['task_dim']
    task_dim=len(next(iter(vectors.values())))
    if task_dim!=config_task_dim:
        raise ValueError(f'Text embedding dimensionality mismatch: {task_dim} vs {config_task_dim}')
    _,model,device=make_model(args,ds[0]['image'].numel(),task_dim,cfg['n_sources'],
                               vectors,meta.get('source_tasks'),meta.get('source_stats'))
    load_checkpoint(path,model=model,device=device)
    model.eval()
    if cli.routing_mode!='semantic':
        if not args.expert_bank:
            raise ValueError('--routing-mode uniform/random requires a K2-B1 expert-bank checkpoint')
        model.set_routing_mode(cli.routing_mode,seed=cli.routing_seed)
    results={}
    predictions=[]
    for name in ('meta_SCUI','meta_UCSI','meta_UCUI'):
        y=[];p=[]
        loader=DataLoader(Subset(ds,splits[name]),batch_size=cli.batch_size,
                          shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
        with torch.inference_mode():
            for batch in loader:
                inp={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
                output=model(inp,None).detach().float().cpu().numpy().ravel()
                target=batch['target'].numpy().ravel()
                y.extend(target.tolist());p.extend(output.tolist())
                if cli.save_predictions:
                    predictions.extend([
                        {'protocol':name,'city':city,'indicator':task,
                         'target':float(yy),'prediction':float(pp)}
                        for city,task,yy,pp in zip(batch['city'],batch['task_name'],target,output)])
        results[name]=summaries(y,p)
        stat=results[name]
        print(f'{Path(path).parent.name:>28} {name:>9} '
              f'n={stat["n"]} mse={stat["mse"]:.3f} r2={stat["r2"]:.3f} '
              f'bias={stat["bias"]:.3f} pearson={stat["pearson"] if stat["pearson"] is not None else "N/A"}',
              flush=True)
    return {
        'checkpoint':str(Path(path).resolve()),
        'epoch':raw['epoch'],
        'heldout_task':meta['heldout_task'],
        'task_encoder':meta['task_encoder'],
        'scale_head':args.scale_head,
        'routing_mode':cli.routing_mode,
        'routing_seed':cli.routing_seed,
        'metrics':results,
    },predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',action='append',required=True,
                        help='Repeat for paired baseline and K2-A checkpoint')
    parser.add_argument('--train-csv')
    parser.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    parser.add_argument('--image-column')
    parser.add_argument('--feature-dir',default='runs/m3_features')
    parser.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    parser.add_argument('--remoteclip-checkpoint')
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--output',default='runs/ktrans_diagnostics')
    parser.add_argument('--save-predictions',action='store_true')
    parser.add_argument('--routing-mode',choices=['semantic','uniform','random'],default='semantic',
                        help='K2-B1 evaluation-only source expert route override')
    parser.add_argument('--routing-seed',type=int,default=42)
    args=parser.parse_args()
    if args.batch_size<1:parser.error('--batch-size must be positive')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    all_results=[]
    for n,path in enumerate(args.checkpoint):
        result,rows=evaluate_checkpoint(path,args)
        result['label']=Path(path).parent.name+('_'+args.routing_mode if args.routing_mode!='semantic' else '')
        all_results.append(result)
        if args.save_predictions:
            name=f'predictions_{n}_{Path(path).parent.name}.csv'
            with (out/name).open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=['protocol','city','indicator','target','prediction'])
                writer.writeheader();writer.writerows(rows)
    (out/'diagnostics.json').write_text(json.dumps(all_results,indent=2))
    with (out/'diagnostics.csv').open('w',newline='') as file:
        fields=['model','epoch','task','protocol','n','target_mean','pred_mean',
                'target_std','pred_std','bias','mse','mae','r2','pearson',
                'oracle_mean_mse','pred_mean_constant_mse','spatial_gain_vs_pred_mean_constant']
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        for result in all_results:
            for protocol,vals in result['metrics'].items():
                writer.writerow({'model':result['label'],'epoch':result['epoch'],
                                 'task':result['heldout_task'],'protocol':protocol,**vals})
    print(f'Wrote {out / "diagnostics.csv"} and {out / "diagnostics.json"}',flush=True)
    print('oracle_mean_mse uses eval labels for DIAGNOSTICS only; never use it as a deployable baseline.')


if __name__=='__main__':main()
