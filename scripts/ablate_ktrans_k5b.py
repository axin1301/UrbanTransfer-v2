"""Inference-only K5-B router/adapter factorial ablation (no training)."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.diagnose_k4_profiles import load_model
from scripts.run_ktrans_k5b import evaluate
from urban_transfer.ktrans_k5b import RegionTaskComposer

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--composer-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--eval-batch-size',type=int,default=1024)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=p.parse_args()
    saved=torch.load(args.composer_checkpoint,map_location='cpu',weights_only=False)
    meta,ds,splits,bank,device=load_model(saved['checkpoint'],args.feature_dir,
                                          args.text_cache_dir,args.image_root)
    from urban_transfer.ktrans_text import qwen_task_vectors
    vectors=qwen_task_vectors(root=args.text_cache_dir,model_id=meta['text_model_id'],
                              profile=meta['task_profile'],only_cached=True)
    model=RegionTaskComposer(bank,vectors,rank=saved['rank']).to(device)
    missing,unexpected=model.load_state_dict(saved['composer'],strict=False)
    allowed=[k for k in missing if k.startswith('bank.')]
    if len(allowed)!=len(missing) or unexpected:
        raise RuntimeError(f'Unexpected composer state keys: {missing} {unexpected}')
    model.eval()
    out=args.output or args.composer_checkpoint.parent/'ablation'
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    original_route=model.route_strength
    original_adapter=model.adapter_strength
    for name,route_on,adapter_on in [
        ('A_k4',False,False),('B_router_only',True,False),
        ('C_adapter_only',False,True),('D_k5b_full',True,True)
    ]:
        model.route_strength=original_route if route_on else 0.
        model.adapter_strength=original_adapter if adapter_on else 0.
        for protocol in ('meta_SCUI','meta_UCUI'):
            scores,w=evaluate(model,ds,splits[protocol],device,args.eval_batch_size)
            rows.append({'variant':name,'protocol':protocol,**scores})
            print(f'{name:<16} {protocol:<10} MSE={scores["mse"]:.3f} '
                  f'R2={scores["r2"]:.3f} Pearson={scores["pearson"]:.3f} '
                  f'Bias={scores["bias"]:+.3f}',flush=True)
    with (out/'evaluation.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (out/'protocol.json').write_text(json.dumps({
        'composer_checkpoint':str(args.composer_checkpoint),
        'base_checkpoint':saved['checkpoint'],
        'heldout_task':meta['heldout_task'],
        'ablation':'inference only; both components jointly trained',
        'scale_router':'original frozen semantic',
        'disclaimer':'DEV meta labels only for evaluation; no refitting'},indent=2))
    print('Saved',out/'evaluation.csv')

if __name__=='__main__':main()
