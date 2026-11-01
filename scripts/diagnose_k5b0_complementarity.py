"""K5-B0: read-only diagnostic for complementarity among nested experts.

Uses K5-A inner last.pt checkpoints; no training or router selection.
All target labels used below are for retrospective diagnosis ONLY, never for
zero-shot routing, validation-time fitting or choosing a deployable model.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.diagnose_k4_profiles import load_model
from scripts.run_ktrans_k5a import extract_outputs,safe_corr

def metrics(y,p):
    return safe_corr(y,p),float(np.mean((p-y)**2))

def diagnose(z,y,sources,pseudo,protocol):
    n=len(sources)
    if z.shape!=(n,len(y)):raise ValueError('Unexpected prediction shape')
    rows=[]
    all_pred=z.mean(axis=0)
    all_r,all_mse=metrics(y,all_pred)
    for j,source in enumerate(sources):
        other=np.delete(z,j,axis=0).mean(axis=0)
        loo_r,loo_mse=metrics(y,other)
        single_r,_=metrics(y,z[j])
        # This is the true marginal contribution of expert j to a
        # uniform ensemble; never use target labels to set inference weights.
        rows.append({'pseudo_target':pseudo,'protocol':protocol,'expert':source,
                     'n':len(y),'single_pearson':single_r,
                     'leave_one_out_pearson':loo_r,
                     'uniform_pearson':all_r,
                     'delta_pearson':all_r-loo_r,
                     'delta_mse':loo_mse-all_mse})
    pair_rows=[]
    for i in range(n):
        for j in range(i+1,n):
            pair_rows.append({'pseudo_target':pseudo,'protocol':protocol,
                'expert_a':sources[i],'expert_b':sources[j],
                'prediction_corr':safe_corr(z[i],z[j]),
                'pair_pearson':safe_corr((z[i]+z[j])/2,y)})
    return rows,pair_rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--k5a-dir',required=True,type=Path)
    p.add_argument('--pseudo-target',default=None)
    p.add_argument('--output',type=Path)
    p.add_argument('--protocol',choices=['meta_SCUI','meta_UCUI'],default='meta_SCUI')
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=p.parse_args()
    info=json.loads((args.k5a_dir/'protocol.json').read_text())
    targets=info['inner_source_tasks']
    if args.pseudo_target and args.pseudo_target not in targets:
        p.error('Pseudo-target must be among inner_source_tasks')
    targets=[args.pseudo_target] if args.pseudo_target else targets
    result_dir=args.output or args.k5a_dir/'complementarity'
    result_dir.mkdir(parents=True,exist_ok=True)
    all_rows=[];all_pairs=[]
    for pseudo in targets:
        ckpt=args.k5a_dir/'inner_models'/pseudo/'last.pt'
        if not ckpt.exists():raise FileNotFoundError(ckpt)
        meta,ds,split,model,device=load_model(ckpt,args.feature_dir,args.text_cache_dir,args.image_root)
        if meta['heldout_task']!=pseudo:raise AssertionError('Pseudo-target mismatch')
        if info['outer_task'] in model.source_tasks:
            raise AssertionError('Outer heldout task leaked into nested expert bank')
        indices=split[args.protocol]
        z,y=extract_outputs(model,ds,indices,device,args.batch_size)
        rows,pairs=diagnose(z,y,list(model.source_tasks),pseudo,args.protocol)
        all_rows+=rows;all_pairs+=pairs
        strongest=max(rows,key=lambda r:r['single_pearson'])
        complementary=max(rows,key=lambda r:r['delta_pearson'])
        print(f'{pseudo}: n={len(y)} uniform_Pearson={rows[0]["uniform_pearson"]:.3f} '
              f'best_single={strongest["expert"]} ({strongest["single_pearson"]:.3f}) '
              f'best_LOO_contribution={complementary["expert"]} '
              f'({complementary["delta_pearson"]:+.3f})',flush=True)
    for filename,rows in [('expert_contributions.csv',all_rows),
                          ('pair_correlations.csv',all_pairs)]:
        if not rows:continue
        with (result_dir/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('\nExpert contribution details (positive delta => including expert helps):')
    for row in all_rows:
        print(f'{row["pseudo_target"]:<30} {row["expert"]:<30} '
              f'single={row["single_pearson"]:.3f} '
              f'delta={row["delta_pearson"]:+.3f} '
              f'pair_MSE_gain={row["delta_mse"]:+.3f}')
    print('Saved to',result_dir)

if __name__=='__main__':main()
