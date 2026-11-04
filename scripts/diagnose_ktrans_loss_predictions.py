"""Read-only paired prediction diagnostics for K6.1 multi-loss gates.

For the same heldout rows, compare transfer-vs-regional spatial headroom,
checkpoint gate displacement, and its realized prediction effect. This is
retrospective DEV diagnosis ONLY: never fit, rank, or select a checkpoint
using these heldout task labels.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from scripts.run_ktrans_k6 import load_components
from urban_transfer.ktrans_k61 import AnchoredRegionalGate,build_source_transfer_routes
from urban_transfer.ktrans_text import qwen_task_vectors

MODES=('mse','corr','rank','combined')

def summary_metrics(gate,global_pred,regional_pred,base_gate,base_pred,pred):
    """Independent numpy-only measurement, with no access to target labels."""
    gate=np.asarray(gate);global_pred=np.asarray(global_pred)
    regional_pred=np.asarray(regional_pred)
    base_gate=np.asarray(base_gate);base_pred=np.asarray(base_pred)
    pred=np.asarray(pred)
    if not (gate.shape==global_pred.shape==regional_pred.shape==
            base_gate.shape==base_pred.shape==pred.shape):
        raise ValueError('Paired prediction vectors must have matching shapes')
    if not len(gate):raise ValueError('Empty partition')
    gap=regional_pred-global_pred
    dg=gate-base_gate
    dp=pred-base_pred
    return {
        'n':len(gate),
        'route_gap_std':float(np.std(gap)),
        'route_gap_mean_abs':float(np.mean(np.abs(gap))),
        'gate_mean':float(np.mean(gate)),
        'gate_std':float(np.std(gate)),
        'gate_delta_mean_abs':float(np.mean(np.abs(dg))),
        'gate_delta_p90_abs':float(np.quantile(np.abs(dg),.9)),
        'pred_delta_mean_abs':float(np.mean(np.abs(dp))),
        'pred_delta_std':float(np.std(dp)),
        'pred_delta_max_abs':float(np.max(np.abs(dp))),
        'identity_error':float(np.max(np.abs(dp-dg*gap))),
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True,
                   help='Parent directory containing k61loss_<task>_<mode> subfolders')
    p.add_argument('--task',required=True,help='Suffix such as education, nighttime, digital_total_cell')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',choices=('meta_SCUI','meta_UCUI'),default='meta_UCUI')
    p.add_argument('--eval-batch-size',type=int,default=1024)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=p.parse_args()
    if args.eval_batch_size<1:p.error('eval-batch-size must be positive')
    saved={}
    for mode in MODES:
        path=args.run_dir/f'k61loss_{args.task}_{mode}'/'gate_last.pt'
        if not path.is_file():p.error(f'Missing completed experiment checkpoint: {path}')
        item=torch.load(path,map_location='cpu',weights_only=False)
        if item.get('loss_mode')!=mode:p.error(f'Checkpoint mode does not match path: {path}')
        if 'gate' not in item:p.error(f'Gate state missing: {path}')
        saved[mode]=item
    ref=saved['mse']
    for mode,item in saved.items():
        for key in ('outer_task','epochs','seed','composer_checkpoint','k5a_protocol'):
            if str(item.get(key))!=str(ref.get(key)):
                p.error(f'{mode} differs from mse in {key}')
    from argparse import Namespace
    load_args=Namespace(composer_checkpoint=Path(ref['composer_checkpoint']),
        k5a_protocol=Path(ref['k5a_protocol']),
        feature_dir=args.feature_dir,text_cache_dir=args.text_cache_dir,
        image_root=args.image_root)
    unified,ds,splits,meta,device,k4=load_components(load_args)
    if meta['heldout_task']!=ref['outer_task']:
        raise AssertionError('Outer heldout task mismatches saved gate checkpoints')
    info=json.loads(load_args.k5a_protocol.read_text())
    vectors=qwen_task_vectors(root=args.text_cache_dir,model_id=meta['text_model_id'],
                         profile=meta['task_profile'],only_cached=True)
    composer=unified.composer
    composer.adapter_strength=0.
    routes=build_source_transfer_routes(vectors,list(composer.source_tasks),
                        meta['source_stats'],info['ridge_weights'],info['route_temperature'])
    gates={}
    for mode in MODES:
        m=AnchoredRegionalGate(composer,info['learned_weights'],routes).to(device)
        m.gate.load_state_dict(saved[mode]['gate'],strict=True)
        m.eval()
        gates[mode]=m
    values={mode:{'gate':[],'pred':[]} for mode in MODES}
    anchor_preds=[];regional_preds=[]
    loader=DataLoader(Subset(ds,splits[args.protocol]),
                      batch_size=args.eval_batch_size,shuffle=False,num_workers=0)
    with torch.inference_mode():
        for batch in loader:
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            _,base_aux=gates['mse'](batch,return_aux=True)
            task_idx=torch.tensor([composer.all_index[t] for t in batch['task_name']],device=device)
            semantic=composer.bank.routing[task_idx]
            mean=semantic@composer.bank.source_means
            std=semantic@composer.bank.source_stds
            a=mean+std*(base_aux['anchor']*base_aux['expert_outputs']).sum(-1)
            regional=mean+std*(base_aux['regional']*base_aux['expert_outputs']).sum(-1)
            anchor_preds.extend(a.cpu().tolist())
            regional_preds.extend(regional.cpu().tolist())
            for mode,gate in gates.items():
                output,aux=gate(batch,return_aux=True)
                # Repeated frozen composer outputs should be identical.
                values[mode]['gate'].extend(aux['gate'].cpu().tolist())
                values[mode]['pred'].extend(output.cpu().tolist())
    base_gate=np.asarray(values['mse']['gate'])
    base_pred=np.asarray(values['mse']['pred'])
    rows=[]
    for mode in MODES:
        row={'loss_mode':mode,'protocol':args.protocol,
             **summary_metrics(values[mode]['gate'],anchor_preds,regional_preds,
                               base_gate,base_pred,values[mode]['pred'])}
        rows.append(row)
        print(f'{mode:<9} route_std={row["route_gap_std"]:.3f} '
             f'gate_delta={row["gate_delta_mean_abs"]:.6f} '
             f'pred_delta={row["pred_delta_mean_abs"]:.6f} '
             f'gate_std={row["gate_std"]:.6f} '
             f'identity_error={row["identity_error"]:.8f}',flush=True)
        if row['identity_error']>1e-4:
            raise AssertionError('Gate mixture identity mismatch; do not interpret results')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'prediction_diagnostics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    (args.output/'protocol.json').write_text(json.dumps({
        'heldout_task':meta['heldout_task'],
        'partition':args.protocol,'k4_checkpoint':str(k4),
        'gate_checkpoints':{mode:str(args.run_dir/f'k61loss_{args.task}_{mode}'/'gate_last.pt') for mode in MODES},
        'evaluation_only':True,'uses_heldout_labels':False,
        'note':'Paired prediction diagnostics, NOT hyperparameter tuning or independent validation.'
    },indent=2))
    print('Saved',args.output/'prediction_diagnostics.csv',flush=True)

if __name__=='__main__':main()
