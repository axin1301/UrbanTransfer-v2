"""K7-G read-only scale/error and source-to-target route-direction audit.

Fixed K7-D routers, frozen K4 experts, identical complete-case rows.
Target labels are accessed only to describe errors, never to select
parameters or refit any model. The affine oracle is IN-SAMPLE ONLY.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_k4_profiles import load_model
from scripts.run_ktrans_k7d import validate_manifest,index_knowledge,vectors_for,extract_expert_batches
from scripts.cache_ktrans_k7d import unique_jobs,merge_records
from scripts.run_ktrans_k7f_gate_scan import fixed_gate_prediction
from urban_transfer.ktrans_k7d import AnchoredKnowledgeRouter,VARIANTS
from urban_transfer.ktrans_k7e import route_response
from urban_transfer.ktrans_k7g import error_decomposition,direction_summary
from urban_transfer.ktrans_k61 import build_source_transfer_routes
from urban_transfer.ktrans_text import qwen_task_vectors

PARTITIONS=('train','meta_SCUI','meta_UCSI','meta_UCUI')
GATES=(('fixed_zero',0.),('trained',None),('fixed_one',1.))

def write_csv(path,rows):
    if not rows:raise ValueError('No diagnostic rows')
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def load_routers(args,bank,device):
    nets={}
    for variant in VARIANTS[1:]:
        path=args.routers_dir/(variant+'_last.pt')
        raw=torch.load(path,map_location='cpu',weights_only=False)
        if raw.get('variant')!=variant or raw.get('seed')!=args.seed or any(
            Path(raw[k]).resolve()!=v.resolve() for k,v in (
               ('k4_checkpoint',args.checkpoint),('k5a_protocol',args.k5a_protocol),
               ('manifest',args.manifest))):
            raise ValueError('K7-D router provenance mismatch: '+str(path))
        net=AnchoredKnowledgeRouter(len(bank.source_tasks)).to(device)
        net.load_state_dict(raw['router'],strict=True)
        net.eval();nets[variant]=net
    return nets

def context(rows,bank,src,target,device):
    n=len(bank.source_tasks)
    ids=[bank.task_to_source.get(row['task']) for row in rows]
    prior=torch.stack([target if j is None else src[j] for j in ids])
    mask=torch.zeros(len(rows),n,dtype=torch.bool,device=device)
    for i,j in enumerate(ids):
        if j is not None:mask[i,j]=True
    semantic=torch.stack([bank.routing[bank.task_to_all[r['task']]].to(device) for r in rows])
    mean=semantic@bank.source_means.to(device)
    std=semantic@bank.source_stds.to(device)
    for i,j in enumerate(ids):
        if j is not None:
            mean[i]=bank.source_means[j]
            std[i]=bank.source_stds[j]
    return prior,mask,mean,std

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','k5a-protocol','manifest','knowledge-cache','routers-dir','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--max-pixels',type=int,default=262144)
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--feature-dir',default='runs/m3_features')
    parser.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    parser.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=parser.parse_args()
    if args.batch_size<1:parser.error('Batch size must be positive')
    meta,ds,splits,bank,device=load_model(args.checkpoint,args.feature_dir,
                                          args.text_cache_dir,args.image_root)
    manifest=json.loads(args.manifest.read_text())
    validate_manifest(manifest,ds,splits,meta,args.checkpoint)
    info=json.loads(args.k5a_protocol.read_text())
    if Path(info['outer_checkpoint']).resolve()!=args.checkpoint.resolve() or info['outer_task']!=meta['heldout_task'] or tuple(info['inner_source_tasks'])!=tuple(bank.source_tasks):
        raise ValueError('K5-A expert ordering/outer target/checkpoint mismatch')
    vectors=qwen_task_vectors(root=args.text_cache_dir,model_id=meta['text_model_id'],
                              profile=meta['task_profile'],only_cached=True)
    src=build_source_transfer_routes(vectors,list(bank.source_tasks),
             meta['source_stats'],info['ridge_weights'],info['route_temperature']).to(device)
    target=torch.as_tensor(info['learned_weights'],dtype=torch.float32,device=device)
    knowledge=index_knowledge(unique_jobs(manifest,args.model,args.max_pixels),
                              merge_records([args.reuse_cache,args.knowledge_cache]))
    nets=load_routers(args,bank,device)
    summaries=[];by_task=[]
    for partition in PARTITIONS:
        rows=[r for r in manifest['rows'] if r['partition']==partition]
        experts,labels=extract_expert_batches(ds,rows,bank,device,args.batch_size)
        experts=experts.to(device)
        y=labels.numpy()
        prior,mask,mean,std=context(rows,bank,src,target,device)
        for variant in VARIANTS[1:]:
            with torch.inference_mode():
                k=vectors_for(rows,knowledge,variant).to(device)
                response=route_response(nets[variant],k,prior,mask,experts,mean,std)
                base=response['baseline'].detach().cpu().numpy()
                for gate_mode,gate in GATES:
                    pred=(response['prediction'] if gate is None else
                          fixed_gate_prediction(response,experts,mean,std,gate)[0])
                    pred=pred.detach().cpu().numpy()
                    description=error_decomposition(y,pred,base)
                    row={'variant':variant,'protocol':partition,
                         'gate_mode':gate_mode,
                         'gate':float(response['gate'].mean()) if gate is None else gate,
                         'role':'source_router_train_IN_SAMPLE' if partition=='train' else 'meta_DEV',
                         **description}
                    summaries.append(row)
                    if partition in ('train','meta_UCSI'):
                        tasks=np.asarray([r['task'] for r in rows])
                        for name in sorted(set(tasks)):
                            idx=np.flatnonzero(tasks==name)
                            sub=error_decomposition(y[idx],pred[idx],base[idx])
                            by_task.append({'variant':variant,'protocol':partition,
                               'gate_mode':gate_mode,'gate':row['gate'],
                               'task':name,'role':row['role'],**sub})
                    print(f'[k7g] {partition:<10} {variant:<11} {gate_mode:<10} '
                          f'MSE={description["mse"]:.3f} '
                          f'dMSE={description["delta_mse_vs_k5a"]:+.3f} '
                          f'dBias2={description["delta_bias_squared"]:+.3f} '
                          f'dCentered={description["delta_centered_mse"]:+.3f}',flush=True)
    direction=[]
    for name in VARIANTS[1:]:
        for gate_mode,_ in GATES:
            source=next(x for x in summaries if x['variant']==name and
                        x['protocol']=='train' and x['gate_mode']==gate_mode)
            for protocol in ('meta_SCUI','meta_UCSI','meta_UCUI'):
                evaluation=next(x for x in summaries if x['variant']==name and
                               x['protocol']==protocol and x['gate_mode']==gate_mode)
                direction.append({'target_protocol':protocol,
                                  **direction_summary(source,evaluation)})
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'error_decomposition.csv',summaries)
    write_csv(args.output/'source_task_decomposition.csv',by_task)
    write_csv(args.output/'transfer_direction.csv',direction)
    (args.output/'protocol.json').write_text(json.dumps({
       'version':'k7g-frozen-error-direction-v1','outer_task':meta['heldout_task'],
       'seed':args.seed,'manifest':str(args.manifest.resolve()),
       'checkpoint':str(args.checkpoint.resolve()),
       'k5a_protocol':str(args.k5a_protocol.resolve()),
       'gate_modes':[name for name,_ in GATES],
       'target_labels_for_diagnostics_only':True,
       'oracle_calibration_status':'SAME-SPLIT IN-SAMPLE DIAGNOSTIC; NOT DEPLOYABLE',
       'source_train_status':'K7-D Router AND K4 Bank have seen source train rows; not cross-fitted',
       'warning':'DEV ONLY: K4 meta-selected, no independent inner crossfitting. Never tune gates or calibration on outer heldout labels.'},indent=2))
    print('[k7g] saved',args.output/'error_decomposition.csv',
          args.output/'transfer_direction.csv',flush=True)

if __name__=='__main__':main()
