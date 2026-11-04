"""K7-H frozen, unlabeled-cohort mean-preserving K7-D counterfactual.

No model optimization or new VLM calls. Preserves K5-A prediction mean
within each predeclared (manifest partition, task) cohort using only
predictions, NOT targets. Labels enter only after correction for DEV
error reporting.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_k4_profiles import load_model
from scripts.run_ktrans_k7d import validate_manifest,index_knowledge,vectors_for,extract_expert_batches
from scripts.cache_ktrans_k7d import unique_jobs,merge_records
from scripts.run_ktrans_k7g_error_diagnostics import load_routers,context
from scripts.run_ktrans_k7f_gate_scan import fixed_gate_prediction
from urban_transfer.ktrans_k7d import VARIANTS
from urban_transfer.ktrans_k7e import route_response
from urban_transfer.ktrans_k7g import error_decomposition
from urban_transfer.ktrans_k7h import preserve_mean,mean_preserving_error_identity
from urban_transfer.ktrans_k61 import build_source_transfer_routes
from urban_transfer.ktrans_text import qwen_task_vectors

MODES=(('fixed_zero',0.),('trained',None),('fixed_one',1.))
PARTITIONS=('train','meta_SCUI','meta_UCSI','meta_UCUI')

def write_csv(path,rows):
    if not rows:raise ValueError('Empty diagnostic CSV')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

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
    meta,ds,splits,bank,device=load_model(a.checkpoint,a.feature_dir,
                                          a.text_cache_dir,a.image_root)
    manifest=json.loads(a.manifest.read_text())
    validate_manifest(manifest,ds,splits,meta,a.checkpoint)
    info=json.loads(a.k5a_protocol.read_text())
    if (Path(info['outer_checkpoint']).resolve()!=a.checkpoint.resolve() or
        info['outer_task']!=meta['heldout_task'] or
        tuple(info['inner_source_tasks'])!=tuple(bank.source_tasks)):
        raise ValueError('K5-A checkpoint/target/source mismatch')
    texts=qwen_task_vectors(root=a.text_cache_dir,model_id=meta['text_model_id'],
                            profile=meta['task_profile'],only_cached=True)
    src=build_source_transfer_routes(texts,list(bank.source_tasks),
        meta['source_stats'],info['ridge_weights'],info['route_temperature']).to(device)
    target=torch.as_tensor(info['learned_weights'],dtype=torch.float32,device=device)
    knowledge=index_knowledge(unique_jobs(manifest,a.model,a.max_pixels),
                              merge_records([a.reuse_cache,a.knowledge_cache]))
    nets=load_routers(a,bank,device)
    results=[];audit=[];per_task=[]
    for partition in PARTITIONS:
        rows=[r for r in manifest['rows'] if r['partition']==partition]
        if not rows:raise ValueError('Empty partition: '+partition)
        tasks=[str(r['task']) for r in rows]
        experts,y=extract_expert_batches(ds,rows,bank,device,a.batch_size)
        experts=experts.to(device)
        labels=y.numpy()
        prior,mask,mean,std=context(rows,bank,src,target,device)
        with torch.inference_mode():
            for variant in VARIANTS[1:]:
                k=vectors_for(rows,knowledge,variant).to(device)
                response=route_response(nets[variant],k,prior,mask,experts,mean,std)
                base=response['baseline'].detach().cpu().numpy()
                for mode,gate in MODES:
                    raw=(response['prediction'] if gate is None else
                         fixed_gate_prediction(response,experts,mean,std,gate)[0])
                    raw=raw.detach().cpu().numpy()
                    fixed,groups=preserve_mean(base,raw,tasks)
                    mean_preserving_error_identity(labels,base,raw,fixed,tasks)
                    for kind,pred in [('original',raw),('mean_preserved',fixed)]:
                        d=error_decomposition(labels,pred,base)
                        record={'variant':variant,'protocol':partition,
                           'gate_mode':mode,
                           'gate':float(response['gate'].mean()) if gate is None else gate,
                           'correction':kind,
                           'role':'router_train_IN_SAMPLE' if partition=='train' else 'meta_DEV',
                           'cohort_policy':'all_manifest_rows_in_same_partition_and_task',
                           'n_cohorts':len(groups),**d}
                        results.append(record)
                        print(f'[k7h] {partition:<10} {variant:<11} {mode:<10} '
                              f'{kind:<14} MSE={d["mse"]:.3f} '
                              f'dMSE={d["delta_mse_vs_k5a"]:+.3f} '
                              f'dBias2={d["delta_bias_squared"]:+.3f} '
                              f'dCentered={d["delta_centered_mse"]:+.3f}',flush=True)
                        for task in sorted(set(tasks)):
                            ix=np.flatnonzero(np.asarray(tasks)==task)
                            sub=error_decomposition(labels[ix],pred[ix],base[ix])
                            per_task.append({'variant':variant,'protocol':partition,
                                 'gate_mode':mode,'correction':kind,'task':task,**sub})
                    audit.extend({'variant':variant,'protocol':partition,
                                  'gate_mode':mode,**item} for item in groups)
    # Pairwise checks: fixed_zero remains identical; labels never used in projection.
    for r in results:
        if r['gate_mode']=='fixed_zero' and abs(r['delta_mse_vs_k5a'])>1e-8:
            raise AssertionError('K5-A fixed-zero must match itself')
    a.output.mkdir(parents=True,exist_ok=True)
    write_csv(a.output/'mean_preservation.csv',results)
    write_csv(a.output/'cohort_audit.csv',audit)
    write_csv(a.output/'per_task_metrics.csv',per_task)
    (a.output/'protocol.json').write_text(json.dumps({
        'version':'k7h-label-free-mean-preservation-v1',
        'seed':a.seed,'k4_checkpoint':str(a.checkpoint.resolve()),
        'k5a_protocol':str(a.k5a_protocol.resolve()),
        'manifest':str(a.manifest.resolve()),
        'routers_dir':str(a.routers_dir.resolve()),
        'cohort_definition':'complete-case manifest rows grouped by (partition, task), all rows jointly used',
        'cohort_selection_uses_labels':False,
        'correction_uses_labels':False,
        'transductive':True,
        'gate_modes':[x[0] for x in MODES],
        'caution':'Posthoc DEV ONLY; full cohort predictions required. No independent K4 inner crossfit. Do not select a correction on UCUI and claim unbiased generalization.'
    },indent=2))
    print('[k7h] saved',a.output/'mean_preservation.csv',flush=True)
if __name__=='__main__':main()
