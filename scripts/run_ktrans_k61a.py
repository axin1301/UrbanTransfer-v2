"""Train K6.1-A gate on source-only pseudo-unseen tasks; assess heldout DEV splits.

K5-A prior, K5-B2 router, K4 experts and scale are frozen. Never use
heldout labels for training/checkpoint selection. The Adapter is OFF.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from scripts.run_ktrans_k6 import load_components,metrics
from urban_transfer.ktrans_k61 import AnchoredRegionalGate,build_source_transfer_routes
from urban_transfer.ktrans_losses import transfer_loss
from urban_transfer.ktrans_text import qwen_task_vectors

def evaluate(gate_model,ds,indices,device,batch_size):
    y=[];pred={'k5a':[],'regional_only':[],'k61a_gate':[]};gates=[]
    gate_model.eval()
    with torch.inference_mode():
        for batch in DataLoader(Subset(ds,indices),batch_size=batch_size,
                                shuffle=False,num_workers=0):
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            result,aux=gate_model(batch,return_aux=True)
            prior=gate_model.composer.bank.routing[
                torch.tensor([gate_model.composer.all_index[t] for t in batch['task_name']],device=device)]
            mean=prior@gate_model.composer.bank.source_means
            std=prior@gate_model.composer.bank.source_stds
            anchor=mean+std*(aux['anchor']*aux['expert_outputs']).sum(-1)
            regional=mean+std*(aux['regional']*aux['expert_outputs']).sum(-1)
            y.extend(batch['target'].cpu().tolist())
            pred['k5a'].extend(anchor.cpu().tolist())
            pred['regional_only'].extend(regional.cpu().tolist())
            pred['k61a_gate'].extend(result.cpu().tolist())
            gates.extend(aux['gate'].cpu().tolist())
    y=np.asarray(y);gate_values=np.asarray(gates)
    return {name:metrics(y,np.asarray(x)) for name,x in pred.items()},{
       'gate_mean':float(gate_values.mean()),
       'gate_std':float(gate_values.std()),
       'gate_p10':float(np.quantile(gate_values,.10)),
       'gate_p90':float(np.quantile(gate_values,.90))}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--composer-checkpoint',required=True,type=Path)
    p.add_argument('--k5a-protocol',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--gate-penalty',type=float,default=.01)
    p.add_argument('--loss-mode',choices=['mse','corr','rank','combined'],default='mse')
    p.add_argument('--corr-weight',type=float,default=.2)
    p.add_argument('--rank-weight',type=float,default=.1)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--eval-batch-size',type=int,default=1024)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=p.parse_args()
    if (args.epochs<=0 or args.lr<=0 or args.gate_penalty<0 or
        args.corr_weight<0 or args.rank_weight<0 or
        args.batch_size<=0 or args.eval_batch_size<=0):
        p.error('Invalid training hyperparameters')
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True,exist_ok=True)
    model,ds,splits,meta,device,k4=load_components(args)
    transfer_info=json.loads(args.k5a_protocol.read_text())
    vectors=qwen_task_vectors(root=args.text_cache_dir,model_id=meta['text_model_id'],
                              profile=meta['task_profile'],only_cached=True)
    composer=model.composer
    composer.adapter_strength=0.
    source_routes=build_source_transfer_routes(
        vectors,list(composer.source_tasks),meta['source_stats'],
        transfer_info['ridge_weights'],transfer_info['route_temperature'])
    gate_model=AnchoredRegionalGate(
        composer,transfer_info['learned_weights'],source_routes).to(device)
    train_rows=ds.df.iloc[splits['train']]
    if meta['heldout_task'] in set(train_rows.indicator_column):
        raise AssertionError('Outer heldout task leaked into source-only gate training')
    if set(train_rows.indicator_column)!=set(composer.source_tasks):
        raise AssertionError('Gate training requires only complete outer source-task set')
    optimizer=torch.optim.AdamW(gate_model.gate.parameters(),lr=args.lr,weight_decay=1e-4)
    for epoch in range(1,args.epochs+1):
        gate_model.train();summed=0.;count=0;avg_gate=0.
        loader=DataLoader(Subset(ds,splits['train']),batch_size=args.batch_size,
            shuffle=True,generator=torch.Generator().manual_seed(args.seed+epoch),num_workers=0)
        for batch in loader:
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            _,aux=gate_model(batch,exclude_own=True,return_aux=True)
            own=torch.tensor([composer.source_index[t] for t in batch['task_name']],device=device)
            label_std=(batch['target']-composer.bank.source_means[own])/composer.bank.source_stds[own]
            loss,parts=transfer_loss(aux['spatial'],label_std,
                batch['task_name'],batch['city'],mode=args.loss_mode,
                corr_weight=args.corr_weight,rank_weight=args.rank_weight)
            loss=loss+args.gate_penalty*aux['gate'].square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n=len(batch['target'])
            summed+=float(loss.detach())*n;avg_gate+=float(aux['gate'].mean().detach())*n;count+=n
        print(f'[k61a] epoch={epoch:02d} source_train_loss={summed/count:.3f} '
              f'gate_mean={avg_gate/count:.3f}',flush=True)
    torch.save({'gate':{k:v.cpu() for k,v in gate_model.gate.state_dict().items()},
                'composer_checkpoint':str(args.composer_checkpoint),
                'k5a_protocol':str(args.k5a_protocol),
                'outer_task':meta['heldout_task'],
                'epochs':args.epochs,'seed':args.seed,
                'gate_penalty':args.gate_penalty,
                'loss_mode':args.loss_mode,'corr_weight':args.corr_weight,
                'rank_weight':args.rank_weight},args.output/'gate_last.pt')
    rows=[]
    for protocol in ('meta_SCUI','meta_UCUI'):
        score,gate_stats=evaluate(gate_model,ds,splits[protocol],device,args.eval_batch_size)
        for variant,met in score.items():
            rows.append({'variant':variant,'protocol':protocol,**met})
            print(f'{protocol:<10} {variant:<16} MSE={met["mse"]:.3f} '
                  f'Pearson={met["pearson"]:.3f} Bias={met["bias"]:+.3f}',flush=True)
        print('gate stats:', ' '.join(f'{k}={v:.3f}' for k,v in gate_stats.items()),flush=True)
    with (args.output/'evaluation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    (args.output/'protocol.json').write_text(json.dumps({
        'outer_task':meta['heldout_task'],'k4_checkpoint':str(k4),
        'composer_checkpoint':str(args.composer_checkpoint),
        'transfer_protocol':str(args.k5a_protocol),
        'train_partition':'outer train only, five source tasks',
        'pseudo_target_policy':'own source expert excluded in both regional and transfer distributions',
        'adapter':'disabled','scale':'frozen semantic',
        'gate_penalty':args.gate_penalty,
        'loss_mode':args.loss_mode,'corr_weight':args.corr_weight,
        'rank_weight':args.rank_weight,
        'gate_init_logit':-2.,'epochs':args.epochs,
        'warning':'Development-only; underlying K4 best.pt previously meta-selected'
    },indent=2))
    print(f'[loss] mode={args.loss_mode} corr_weight={args.corr_weight:.3f} '
          f'rank_weight={args.rank_weight:.3f}',flush=True)
    print('Saved',args.output/'evaluation.csv',flush=True)

if __name__=='__main__':main()
