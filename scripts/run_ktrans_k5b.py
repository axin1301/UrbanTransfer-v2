"""K5-B pilot: train region-conditioned composer over frozen K4 expert bank.

Only outer source-task/seen-city rows are used for training; outer heldout
task labels are used for reporting DEV metrics only. The training loss is
standardized source-target MSE with the source's own expert masked.
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
from urban_transfer.ktrans_text import qwen_task_vectors

def evaluate(model,ds,indices,device,batch_size):
    preds=[];ys=[];w=[]
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(Subset(ds,indices),batch_size=batch_size,num_workers=0):
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            prediction,aux=model(batch,return_aux=True)
            preds.extend(prediction.cpu().tolist())
            ys.extend(batch['target'].cpu().tolist())
            w.extend(aux['weights'].cpu().tolist())
    pred=np.asarray(preds);y=np.asarray(ys)
    mse=float(np.mean((pred-y)**2));var=float(np.var(y))
    return {'n':len(y),'mse':mse,'r2':1.-mse/var if var>1e-12 else float('nan'),
            'pearson':safe_corr(pred,y),'bias':float(np.mean(pred-y)),
            'pred_std':float(pred.std())},np.asarray(w).mean(axis=0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--eval-batch-size',type=int,default=1024)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.rank<1 or args.lr<=0:p.error('positive hyperparameters required')
    torch.manual_seed(args.seed)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    meta,ds,splits,bank,device=load_model(args.checkpoint,args.feature_dir,args.text_cache_dir,args.image_root)
    vectors=qwen_task_vectors(root=args.text_cache_dir,model_id=meta['text_model_id'],
        profile=meta['task_profile'],only_cached=True)
    model=RegionTaskComposer(bank,vectors,rank=args.rank).to(device)
    opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=args.lr,weight_decay=1e-4)
    train_rows=ds.df.iloc[splits['train']]
    if meta['heldout_task'] in set(train_rows.indicator_column):
        raise AssertionError('Outer heldout task leaked into K5-B training')
    if set(train_rows.indicator_column)!=set(bank.source_tasks):
        raise AssertionError('Unexpected outer source tasks')
    baseline={}
    for split in ('meta_SCUI','meta_UCUI'):
        result,_=evaluate(model,ds,splits[split],device,args.eval_batch_size)
        baseline[split]=result
    for epoch in range(1,args.epochs+1):
        model.train();s=0.;n=0
        gen=torch.Generator().manual_seed(args.seed+epoch)
        loader=DataLoader(Subset(ds,splits['train']),batch_size=args.batch_size,
                          shuffle=True,generator=gen,num_workers=0)
        for batch in loader:
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            pred,aux=model(batch,exclude_own=True,return_aux=True)
            indices=torch.tensor([bank.task_to_source[t] for t in batch['task_name']],device=device)
            target_std=(batch['target']-bank.source_means[indices])/bank.source_stds[indices]
            # Standardized label loss avoids TMI/Population scale imbalance;
            # the own-expert mask forces transfer via alternative source experts.
            loss=((aux['spatial']-target_std)**2).mean()
            # Mild adapter regularization; no heldout labels in training.
            loss=loss+1e-4*(aux['residual']**2).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            opt.step();s+=float(loss.detach())*len(batch['target']);n+=len(batch['target'])
        print(f'[k5b] epoch={epoch:02d} source_only_train_loss={s/n:.3f}',flush=True)
    torch.save({'composer':{k:v.cpu() for k,v in model.state_dict().items()
                             if not k.startswith('bank.')},
                'checkpoint':str(args.checkpoint),
                'epoch':args.epochs,'seed':args.seed,'rank':args.rank},
               out/'composer.pt')
    rows=[]
    for split in ('meta_SCUI','meta_UCUI'):
        after,weights=evaluate(model,ds,splits[split],device,args.eval_batch_size)
        for name,met in [('k4_initial',baseline[split]),('k5b',after)]:
            rows.append({'route':name,'protocol':split,**met})
        print(f'  {split}: K4 MSE={baseline[split]["mse"]:.3f} '
              f'Pearson={baseline[split]["pearson"]:.3f} '
              f'| K5-B MSE={after["mse"]:.3f} Pearson={after["pearson"]:.3f} '
              f'Bias={after["bias"]:+.3f}',flush=True)
        print('  mean route:', ' '.join(f'{t}={w:.3f}' for t,w in zip(bank.source_tasks,weights)),flush=True)
    with (out/'evaluation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'protocol.json').write_text(json.dumps({'base_checkpoint':str(args.checkpoint),
        'heldout_task':meta['heldout_task'],'heldout_cities':meta['heldout_cities'],
        'training_split':'outer train only','own_source_expert_masked':True,
        'scale_routing':'frozen semantic','epochs':args.epochs,
        'warning':'Development-only: K4 best.pt previously chosen by meta validation'},indent=2))
    print('Saved',out/'evaluation.csv')

if __name__=='__main__':main()
