"""K7-F: fixed-gate *posthoc* counterfactuals on frozen K7-D candidates.

Never trains/updates a model; never uses heldout labels to select gate.
Same complete-case rows and K4/K5-A frozen experts for all variants.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_k4_profiles import load_model
from scripts.run_ktrans_k7d import validate_manifest,index_knowledge,vectors_for,extract_expert_batches
from scripts.cache_ktrans_k7d import unique_jobs,merge_records
from scripts.run_ktrans_k6 import metrics
from urban_transfer.ktrans_k7d import AnchoredKnowledgeRouter,VARIANTS
from urban_transfer.ktrans_k7e import route_response
from urban_transfer.ktrans_k61 import build_source_transfer_routes
from urban_transfer.ktrans_text import qwen_task_vectors

GATES=(0.,0.05,0.118,0.25,0.5,0.75,1.)

def fixed_gate_prediction(response,expert,mean,std,gate):
    if not 0<=gate<=1:raise ValueError('Invalid fixed gate')
    w=(1.-gate)*response['prior']+gate*response['candidate']
    return mean+std*(w*expert).sum(-1),w

def evaluate_grid(response,expert,mean,std,labels,variant,protocol,gates=GATES):
    results=[]
    for gate in gates:
        pred,w=fixed_gate_prediction(response,expert,mean,std,gate)
        pred=pred.detach().cpu().numpy()
        score=metrics(labels,pred)
        shift=(w-response['prior']).abs().sum(-1)
        results.append({'variant':variant,'protocol':protocol,'gate_mode':'fixed',
                        'gate':gate,**score,
                        'route_prior_l1_mean':float(shift.mean()),
                        'prediction_delta_abs_mean':float(np.abs(pred-response['baseline'].detach().cpu().numpy()).mean())})
    learned=response['prediction'].detach().cpu().numpy()
    score=metrics(labels,learned)
    results.append({'variant':variant,'protocol':protocol,'gate_mode':'trained',
                    'gate':float(response['gate'].mean()),**score,
                    'route_prior_l1_mean':float((response['final']-response['prior']).abs().sum(-1).mean()),
                    'prediction_delta_abs_mean':float(np.abs(learned-response['baseline'].detach().cpu().numpy()).mean())})
    base=results[0]['mse']
    for row in results:row['delta_mse_vs_fixed_zero']=row['mse']-base
    return results

def save_csv(path,rows):
    if not rows:raise ValueError('No diagnostics')
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','k5a-protocol','manifest','knowledge-cache','routers-dir','output'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    a=p.parse_args()
    if a.batch_size<1:p.error('Invalid batch size')
    meta,ds,splits,bank,device=load_model(a.checkpoint,a.feature_dir,a.text_cache_dir,a.image_root)
    manifest=json.loads(a.manifest.read_text())
    validate_manifest(manifest,ds,splits,meta,a.checkpoint)
    info=json.loads(a.k5a_protocol.read_text())
    if Path(info['outer_checkpoint']).resolve()!=a.checkpoint.resolve() or info['outer_task']!=meta['heldout_task'] or tuple(info['inner_source_tasks'])!=tuple(bank.source_tasks):
        raise ValueError('K5-A checkpoint/task/source mismatch')
    vectors=qwen_task_vectors(root=a.text_cache_dir,model_id=meta['text_model_id'],
                              profile=meta['task_profile'],only_cached=True)
    src=build_source_transfer_routes(vectors,list(bank.source_tasks),
         meta['source_stats'],info['ridge_weights'],info['route_temperature']).to(device)
    target=torch.as_tensor(info['learned_weights'],dtype=torch.float32,device=device)
    knowledge=index_knowledge(unique_jobs(manifest,a.model,a.max_pixels),
                              merge_records([a.reuse_cache,a.knowledge_cache]))
    nets={}
    for variant in VARIANTS[1:]:
        file=a.routers_dir/(variant+'_last.pt')
        raw=torch.load(file,map_location='cpu',weights_only=False)
        if raw.get('variant')!=variant or raw.get('seed')!=a.seed or any(
            Path(raw[k]).resolve()!=path.resolve() for k,path in (
            ('k4_checkpoint',a.checkpoint),('k5a_protocol',a.k5a_protocol),
            ('manifest',a.manifest))):
            raise ValueError('K7-D router provenance mismatch: '+str(file))
        net=AnchoredKnowledgeRouter(len(bank.source_tasks)).to(device)
        net.load_state_dict(raw['router'],strict=True)
        net.eval();nets[variant]=net
    results=[]
    for protocol in ('meta_SCUI','meta_UCSI','meta_UCUI'):
        rows=[r for r in manifest['rows'] if r['partition']==protocol]
        expert,labels=extract_expert_batches(ds,rows,bank,device,a.batch_size)
        expert=expert.to(device)
        own=[bank.task_to_source.get(r['task']) for r in rows]
        prior=torch.stack([target if i is None else src[i] for i in own])
        mask=torch.zeros(len(rows),len(bank.source_tasks),dtype=torch.bool,device=device)
        for i,j in enumerate(own):
            if j is not None:mask[i,j]=True
        semantic=torch.stack([bank.routing[bank.task_to_all[r['task']]].to(device) for r in rows])
        mean=semantic@bank.source_means.to(device)
        std=semantic@bank.source_stds.to(device)
        for i,j in enumerate(own):
            if j is not None:
                mean[i]=bank.source_means[j];std[i]=bank.source_stds[j]
        with torch.inference_mode():
            for variant in VARIANTS[1:]:
                k=vectors_for(rows,knowledge,variant).to(device)
                response=route_response(nets[variant],k,prior,mask,expert,mean,std)
                part=evaluate_grid(response,expert,mean,std,labels.numpy(),variant,protocol)
                results.extend(part)
                for row in part:
                    print(f'[k7f] {protocol:<10} {variant:<11} '
                         f'{row["gate_mode"]:<7} gate={row["gate"]:.3f} '
                         f'MSE={row["mse"]:.3f} Pearson={row["pearson"]:.3f} '
                         f'deltaMSE={row["delta_mse_vs_fixed_zero"]:+.3f}',flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    save_csv(a.output/'gate_scan.csv',results)
    (a.output/'protocol.json').write_text(json.dumps({
        'version':'k7f-posthoc-fixed-gate-v1','gate_grid':list(GATES),
        'seed':a.seed,'manifest':str(a.manifest.resolve()),
        'checkpoint':str(a.checkpoint.resolve()),
        'k5a_protocol':str(a.k5a_protocol.resolve()),
        'routers_dir':str(a.routers_dir.resolve()),
        'optimization_from_heldout_labels':False,
        'warning':'POSTHOC DEV ONLY: gate grid is a diagnostic, never tune on meta_UCUI and claim independent test improvement. K4 meta-selected; not strictly inner crossfitted.'
    },indent=2))
    print('[k7f] saved',a.output/'gate_scan.csv',flush=True)
if __name__=='__main__':main()
