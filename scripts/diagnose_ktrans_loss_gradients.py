"""K6.1 read-only source-train diagnostics: coverage and gate gradients.

No optimizer or training. Uses the SAME deterministic source-only batches
for all four saved loss-mode checkpoints and never reads heldout-task labels.
Measures weighted auxiliary gradient magnitude and conflict against MSE.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader,Subset
from scripts.run_ktrans_k6 import load_components
from urban_transfer.ktrans_k61 import AnchoredRegionalGate,build_source_transfer_routes
from urban_transfer.ktrans_text import qwen_task_vectors
from urban_transfer.ktrans_losses import grouped_correlation_loss,city_pairwise_rank_loss

MODES=('mse','corr','rank','combined')
def group_coverage(target,tasks,cities,min_size=8,margin=.2,max_pairs=256):
    """Replicate the grouping, label gap and pair cap from ktrans_losses."""
    groups={}
    for t in set(tasks):
        ix=[i for i,s in enumerate(tasks) if s==t]
        if len(ix)>=min_size:
            y=target[ix]
            if float(torch.linalg.vector_norm(y-y.mean()))>=1e-6:
                groups[t]=len(ix)
    city_groups={}
    for i,key in enumerate(zip(tasks,cities)):
        city_groups.setdefault(key,[]).append(i)
    pairs=0;rank_groups=0
    for ix in city_groups.values():
        if len(ix)<2:continue
        order=sorted(ix,key=lambda i:float(target[i]))
        half=len(order)//2
        lo=order[:half];hi=order[-half:]
        valid=sum(float(target[j]-target[i])>margin for i,j in list(zip(lo,hi))[:max_pairs])
        if valid:
            pairs+=valid;rank_groups+=1
    return {'corr_groups':len(groups),'corr_samples':sum(groups.values()),
            'rank_groups':rank_groups,'rank_pairs':pairs}

def gradient(loss,params):
    if loss is None or not loss.requires_grad:
        return torch.zeros(sum(x.numel() for x in params),device=params[0].device)
    grad=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                      for p,g in zip(params,grad)])

def norm(x):
    return float(torch.linalg.vector_norm(x).detach().cpu())

def cosine(a,b,eps=1e-12):
    na=norm(a);nb=norm(b)
    return float((a@b).detach().cpu()/(na*nb)) if na>eps and nb>eps else float('nan')

def analyze_batch(model,batch,corr_weight=.2,rank_weight=.1):
    """Unweighted and optimizer-weighted gradients, source-training rows only."""
    composer=model.composer
    _,aux=model(batch,exclude_own=True,return_aux=True)
    own=torch.tensor([composer.source_index[t] for t in batch['task_name']],
                     device=batch['target'].device)
    target=(batch['target']-composer.bank.source_means[own])/composer.bank.source_stds[own]
    spatial=aux['spatial']
    losses={'mse':(spatial-target).square().mean(),
            'corr':grouped_correlation_loss(spatial,target,batch['task_name']),
            'rank':city_pairwise_rank_loss(spatial,target,batch['task_name'],batch['city'])}
    params=list(model.gate.parameters())
    grads={name:gradient(value,params) for name,value in losses.items()}
    cover=group_coverage(target.detach().cpu(),batch['task_name'],batch['city'])
    delta=aux['regional']-aux['anchor']
    expert_delta=(delta*aux['expert_outputs']).sum(-1)
    rows={**cover,'n':len(target),
          'gate_mean':float(aux['gate'].detach().mean()),
          'gate_std':float(aux['gate'].detach().std(unbiased=False)),
          'route_l1_mean':float(delta.detach().abs().sum(-1).mean()),
          'expert_delta_abs_mean':float(expert_delta.detach().abs().mean())}
    for name,value in losses.items():
        rows[name+'_value']=float(value.detach())
        rows[name+'_grad_norm']=norm(grads[name])
    for name,weight in (('corr',corr_weight),('rank',rank_weight)):
        rows[name+'_weighted_grad_norm']=weight*norm(grads[name])
        rows[name+'_cos_mse']=cosine(grads['mse'],grads[name])
    rows['weighted_corr_to_mse']=(
        rows['corr_weighted_grad_norm']/(rows['mse_grad_norm']+1e-12))
    rows['weighted_rank_to_mse']=(
        rows['rank_weighted_grad_norm']/(rows['mse_grad_norm']+1e-12))
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-prefix',default='runs/k61loss_education')
    p.add_argument('--output',type=Path,default=Path('runs/k61loss_education_diagnostics'))
    p.add_argument('--batches',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=p.parse_args()
    if args.batches<1 or args.batch_size<1:p.error('Positive batches/batch-size required')
    checkpoints={}
    for mode in MODES:
        path=Path(f'{args.runs_prefix}_{mode}')/'gate_last.pt'
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        if ckpt.get('loss_mode','mse')!=mode:
            raise ValueError(f'Loss mode mismatch in {path}')
        checkpoints[mode]=ckpt
    first=checkpoints['mse']
    for name,saved in checkpoints.items():
        for key in ('composer_checkpoint','k5a_protocol','outer_task','epochs','seed',
                    'gate_penalty','corr_weight','rank_weight'):
            if saved.get(key)!=first.get(key):
                raise ValueError(f'Cannot compare runs: {name} differs in {key}')
    if args.seed!=first['seed']:
        raise ValueError('Batch seed must match saved training seed')
    from types import SimpleNamespace
    cfg=SimpleNamespace(composer_checkpoint=Path(first['composer_checkpoint']),
                        k5a_protocol=Path(first['k5a_protocol']),
                        feature_dir=args.feature_dir,text_cache_dir=args.text_cache_dir,
                        image_root=args.image_root)
    unified,ds,splits,meta,device,_=load_components(cfg)
    if meta['heldout_task']!=first['outer_task']:
        raise ValueError('Heldout task differs from saved runs')
    train_df=ds.df.iloc[splits['train']]
    if meta['heldout_task'] in set(train_df.indicator_column):
        raise AssertionError('Heldout task labels entered diagnostic training partition')
    info=json.loads(cfg.k5a_protocol.read_text())
    vectors=qwen_task_vectors(root=cfg.text_cache_dir,model_id=meta['text_model_id'],
                              profile=meta['task_profile'],only_cached=True)
    composer=unified.composer
    composer.adapter_strength=0.
    source_routes=build_source_transfer_routes(vectors,list(composer.source_tasks),
            meta['source_stats'],info['ridge_weights'],info['route_temperature'])
    loader=DataLoader(Subset(ds,splits['train']),batch_size=args.batch_size,
        shuffle=True,generator=torch.Generator().manual_seed(args.seed+1),num_workers=0)
    # Deterministic preselected source-only batches; shared across all modes.
    batches=[]
    for batch in loader:
        batches.append({k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()})
        if len(batches)>=args.batches:break
    if not batches:raise ValueError('Empty outer source training partition')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for mode in MODES:
        model=AnchoredRegionalGate(composer,info['learned_weights'],source_routes).to(device)
        model.gate.load_state_dict(checkpoints[mode]['gate'])
        model.eval()
        for i,batch in enumerate(batches,1):
            details=analyze_batch(model,batch,
                  corr_weight=first['corr_weight'],rank_weight=first['rank_weight'])
            rows.append({'loss_mode':mode,'batch':i,**details})
        summary=[x for x in rows if x['loss_mode']==mode]
        def avg(key):
            return sum(x[key] for x in summary)/len(summary)
        print(f'{mode:<9} gate_mean={avg("gate_mean"):.3f} '
              f'corr_groups={avg("corr_groups"):.1f} '
              f'rank_pairs={avg("rank_pairs"):.1f} '
              f'grad_mse={avg("mse_grad_norm"):.6f} '
              f'corr/mse={avg("weighted_corr_to_mse"):.3f} '
              f'rank/mse={avg("weighted_rank_to_mse"):.3f}',flush=True)
    with (args.output/'batch_gradients.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    keys=['gate_mean','gate_std','route_l1_mean','expert_delta_abs_mean',
          'corr_groups','corr_samples','rank_groups','rank_pairs',
          'mse_value','corr_value','rank_value','mse_grad_norm',
          'corr_weighted_grad_norm','rank_weighted_grad_norm',
          'corr_cos_mse','rank_cos_mse','weighted_corr_to_mse','weighted_rank_to_mse']
    summary=[]
    for mode in MODES:
        records=[x for x in rows if x['loss_mode']==mode]
        summary.append({'loss_mode':mode,**{
            key:sum(float(x[key]) for x in records)/len(records) for key in keys}})
    with (args.output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    (args.output/'protocol.json').write_text(json.dumps({
        'run_prefix':args.runs_prefix,'source_partition':'outer train only',
        'outer_task':meta['heldout_task'],'diagnostic_batches':len(batches),
        'batch_size':args.batch_size,'seed':args.seed,
        'mode_comparison':'identical deterministic source-only batches',
        'gradient_scope':'gate parameters only; frozen K4 and K5-B2',
        'note':'Read-only diagnostics; no optimizer, no outer heldout labels'
    },indent=2))
    print('Saved',args.output/'summary.csv',args.output/'batch_gradients.csv',flush=True)

if __name__=='__main__':main()
