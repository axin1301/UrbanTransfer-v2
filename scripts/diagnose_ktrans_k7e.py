"""K7-E inference-only response diagnostics. No new VLM or router training."""
import argparse,csv,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_k4_profiles import load_model
from scripts.run_ktrans_k7d import validate_manifest,index_knowledge,vectors_for,extract_expert_batches
from scripts.cache_ktrans_k7d import unique_jobs,merge_records
from urban_transfer.ktrans_k7d import AnchoredKnowledgeRouter,VARIANTS
from urban_transfer.ktrans_k7e import route_response,summarise,expert_rows
from urban_transfer.ktrans_k61 import build_source_transfer_routes
from urban_transfer.ktrans_text import qwen_task_vectors

def save_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
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
    jobs=unique_jobs(manifest,a.model,a.max_pixels)
    knowledge=index_knowledge(jobs,merge_records([a.reuse_cache,a.knowledge_cache]))
    nets={'k5a':None}
    for variant in VARIANTS[1:]:
        path=a.routers_dir/(variant+'_last.pt')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['variant']!=variant or state['seed']!=a.seed or any(
            Path(state[key]).resolve()!=expected.resolve() for key,expected in (
                ('k4_checkpoint',a.checkpoint),('k5a_protocol',a.k5a_protocol),('manifest',a.manifest))):
            raise ValueError('Router provenance mismatch: '+str(path))
        net=AnchoredKnowledgeRouter(len(bank.source_tasks)).to(device)
        net.load_state_dict(state['router'],strict=True);net.eval();nets[variant]=net
    summary=[];expert_rows_all=[]
    for split in ('meta_SCUI','meta_UCSI','meta_UCUI'):
        rows=[r for r in manifest['rows'] if r['partition']==split]
        expert,_=extract_expert_batches(ds,rows,bank,device,a.batch_size)
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
            for variant in VARIANTS:
                k=torch.zeros(len(rows),36,device=device) if variant=='k5a' else vectors_for(rows,knowledge,variant).to(device)
                response=route_response(nets[variant],k,prior,mask,expert,mean,std)
                d={'variant':variant,'protocol':split,**summarise(k,response,expert,[r['task'] for r in rows])}
                summary.append(d)
                expert_rows_all.extend({'variant':variant,'protocol':split,**x}
                    for x in expert_rows(response,bank.source_tasks))
                print(f'[k7e] {split} {variant} knowledge_std={d["knowledge_mean_feature_std"]:.6f} '
                      f'candidate_L1={d["candidate_prior_l1_mean"]:.6f} '
                      f'final_L1={d["final_prior_l1_mean"]:.6f} '
                      f'prediction_delta_std={d["prediction_delta_std"]:.6f}',flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    save_csv(a.output/'summary.csv',summary)
    save_csv(a.output/'expert_routes.csv',expert_rows_all)
    (a.output/'protocol.json').write_text(json.dumps({
       'version':'k7e-frozen-v1','seed':a.seed,'manifest':str(a.manifest.resolve()),
       'checkpoint':str(a.checkpoint.resolve()),'k5a_protocol':str(a.k5a_protocol.resolve()),
       'heldout_task_optimized':False,
       'warning':'DEV ONLY: K4 meta-selected, not independently cross-fitted. No VLM inference or router updates.'},indent=2))
    print('[k7e] saved',a.output/'summary.csv',flush=True)
if __name__=='__main__':main()
