"""Single-fold K6 evaluation with a fixed 8-variant ablation matrix.

All trainable weights are loaded from pre-existing source-only checkpoints;
the outer heldout labels are used ONLY for reporting development metrics.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from scripts.diagnose_k4_profiles import load_model
from scripts.run_ktrans_k5a import safe_corr
from urban_transfer.ktrans_k5b import RegionTaskComposer
from urban_transfer.ktrans_k6 import UnifiedKTransMoE,VARIANTS
from urban_transfer.ktrans_text import qwen_task_vectors

def metrics(y,p):
    err=p-y;mse=float(np.mean(err**2));var=float(np.var(y))
    return {'n':len(y),'mse':mse,'rmse':float(np.sqrt(mse)),
        'mae':float(np.mean(np.abs(err))),
        'r2':float(1-mse/var) if var>1e-12 else float('nan'),
        'pearson':safe_corr(y,p),'bias':float(np.mean(err)),
        'pred_std':float(np.std(p))}

def load_components(args):
    saved=torch.load(args.composer_checkpoint,map_location='cpu',weights_only=False)
    if saved.get('version')!='k5b2' or saved.get('adapter_cap') is None:
        raise ValueError('K6 requires the stabilized K5-B2 composer checkpoint')
    k4=Path(saved['checkpoint']).resolve()
    meta,ds,splits,bank,device=load_model(k4,args.feature_dir,
                                           args.text_cache_dir,args.image_root)
    transfer_info=json.loads(args.k5a_protocol.read_text())
    if Path(transfer_info['outer_checkpoint']).resolve()!=k4:
        raise ValueError('K5-A and K5-B2 use different outer K4 checkpoints')
    if transfer_info['outer_task']!=meta['heldout_task']:
        raise ValueError('K5-A target differs from K5-B2 heldout task')
    if tuple(transfer_info['inner_source_tasks'])!=bank.source_tasks:
        raise ValueError('K5-A source expert ordering differs')
    semantic=bank.routing[bank.task_to_all[meta['heldout_task']]].cpu().numpy()
    if not np.allclose(transfer_info['semantic_weights'],semantic,atol=1e-5):
        raise ValueError('K5-A frozen semantic weights differ from this checkpoint')
    vectors=qwen_task_vectors(root=args.text_cache_dir,model_id=meta['text_model_id'],
                              profile=meta['task_profile'],only_cached=True)
    composer=RegionTaskComposer(bank,vectors,rank=saved['rank'],
                                 adapter_cap=saved['adapter_cap']).to(device)
    missing,unexpected=composer.load_state_dict(saved['composer'],strict=False)
    if any(not k.startswith('bank.') for k in missing) or unexpected:
        raise ValueError(f'Unexpected K5-B2 state keys {missing}, {unexpected}')
    composer.eval()
    model=UnifiedKTransMoE(composer,vectors,meta['source_stats'],
                            meta['heldout_task'],transfer_info['learned_weights']).to(device)
    return model,ds,splits,meta,device,k4

def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    model,ds,splits,meta,device,k4=load_components(args)
    variants=list(VARIANTS)
    rows=[]
    for protocol in ('meta_SCUI','meta_UCSI','meta_UCUI'):
        y=[];preds={name:[] for name in variants}
        with torch.inference_mode():
            for batch in DataLoader(Subset(ds,splits[protocol]),
                       batch_size=args.eval_batch_size,shuffle=False,num_workers=0):
                batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
                y.extend(batch['target'].cpu().numpy().tolist())
                for name in variants:
                    p=model.forward_variant(batch,name)
                    preds[name].extend(p.cpu().numpy().tolist())
        y=np.asarray(y)
        for name in variants:
            row={'outer_task':meta['heldout_task'],'variant':name,'protocol':protocol,
                 **metrics(y,np.asarray(preds[name]))}
            rows.append(row)
        print(f'[k6] {meta["heldout_task"]} {protocol} '
              + ' '.join(f'{name}: MSE={next(r for r in rows if r["variant"]==name and r["protocol"]==protocol)["mse"]:.3f}'
                  for name in ('k4','k5a','k5b2','k6_full')),flush=True)
    with (args.output/'evaluation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'outer_task':meta['heldout_task'],
             'source_tasks':list(model.bank.source_tasks),
             'k4_checkpoint':str(k4),
             'composer_checkpoint':str(args.composer_checkpoint.resolve()),
             'transfer_protocol':str(args.k5a_protocol.resolve()),
             'variant_definitions':{k:{'spatial_route':v[0],'adapter':v[1],'scale':v[2]}
                                    for k,v in VARIANTS.items()},
             'k5c_moments':model.k5c_moments,
             'k5d_moments':model.k5d_moments,'k5d_alpha':model.k5d.alpha,
             'evaluation_note':'Development meta partitions; no outer target labels in model fitting; underlying K4 best.pt was meta-selected'}
    (args.output/'protocol.json').write_text(json.dumps(summary,indent=2))
    print('Saved',args.output/'evaluation.csv',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--composer-checkpoint',type=Path,required=True)
    p.add_argument('--k5a-protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--eval-batch-size',type=int,default=1024)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=p.parse_args()
    if args.eval_batch_size<1:p.error('Batch size must be positive')
    run(args)

if __name__=='__main__':main()
