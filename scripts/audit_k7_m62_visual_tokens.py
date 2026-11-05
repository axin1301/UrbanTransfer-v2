"""Read-only M6.2 visual-token intervention on a FROZEN trained checkpoint.

Keep RemoteCLIP, geo context, Education definition and relation tokens
exactly unchanged. Intervene ONLY on the Qwen-VL visual token positions:
matched / circularly swapped from a different image / all-zero.
Pseudo-dev Education labels are used for scoring, never optimization.
No Qwen-VL forwards, new fits, or official test files.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from scripts.run_k7_m61_student import load_precomputed
from scripts.run_k7_m62_task_query import validate_split
from scripts.run_z1_pilot import scores
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.k7_mllm_hidden import HiddenStore
from urban_transfer.k7_m62_task_query import TaskQueriedRouter


def replace_visual_only(original,other,n_visual,mode):
    """Do not mutate cached originals. Text token IDs are never replaced."""
    if original.ndim!=3 or original.shape[1]<=n_visual or n_visual<1:
        raise ValueError("Expected structured [batch, visual+text, dimension] tokens")
    if mode=="matched":
        return original.clone()
    changed=original.clone()
    if mode=="zero_visual":
        changed[:,:n_visual,:]=0.
    elif mode=="swap_visual":
        if other is None or other.shape!=original.shape:
            raise ValueError("Swap requires matched batch-shaped donor tokens")
        changed[:,:n_visual,:]=other[:,:n_visual,:]
    else:
        raise ValueError("Unknown visual token intervention: "+mode)
    return changed


def summary(values):
    a=np.asarray(values,dtype=np.float64).reshape(-1)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Empty/nonfinite visual audit values")
    return {"mean":float(a.mean()),"std":float(a.std()),
            "min":float(a.min()),"max":float(a.max())}


@torch.inference_mode()
def evaluate_visual_interventions(model,x,ctx,y,indices,cache,
                                  n_visual,mean,scale,batch_size,device):
    """Compute full-partition interventions with globally shifted donor indices.

    A full split rotation (not per-mini-batch rotation) guarantees each
    swapped donor is another pseudo-dev image in the same partition,
    even when batch size=1.
    """
    indices=np.asarray(indices,dtype=np.int64)
    if len(indices)<2 or len(set(indices.tolist()))!=len(indices):
        raise ValueError("Need at least two unique partition rows")
    if scale<=0:
        raise ValueError("Target prediction scale must be positive")
    modes=("matched","swap_visual","zero_visual")
    predictions={mode:[] for mode in modes}
    routes={mode:[] for mode in modes}
    queries={mode:[] for mode in modes}
    for start in range(0,len(indices),batch_size):
        ix=indices[start:start+batch_size]
        shifted=indices[(np.arange(start,start+len(ix))+1)%len(indices)]
        original=torch.stack([cache[int(i),HELDOUT_TASK] for i in ix]).to(device)
        donors=torch.stack([cache[int(i),HELDOUT_TASK] for i in shifted]).to(device)
        x_batch=x[ix].to(device)
        ctx_batch=ctx[ix].to(device)
        for mode in modes:
            h=replace_visual_only(original,donors,n_visual,mode)
            pred,w,task,_=model(x_batch,ctx_batch,h,return_aux=True)
            predictions[mode].append(pred.cpu().numpy())
            routes[mode].append(w[:,0].cpu().numpy())
            queries[mode].append(task.cpu())
    pred={mode:np.concatenate(v) for mode,v in predictions.items()}
    weight={mode:np.concatenate(v) for mode,v in routes.items()}
    query={mode:torch.cat(v) for mode,v in queries.items()}
    # The task Query depends on definition+relation ONLY. This equality
    # is an essential intervention sanity check, not a research result.
    for mode in modes[1:]:
        if not torch.allclose(query["matched"],query[mode],
                              atol=1e-6,rtol=1e-6):
            raise RuntimeError("Visual intervention unexpectedly changed Task Query")
    target=y[torch.from_numpy(indices.copy())].numpy()
    report={}
    anchor=pred["matched"]
    base_route=weight["matched"]
    for mode in modes:
        values=mean+scale*pred[mode]
        report[mode]={
            **scores(values,target),
            "population_weight":summary(weight[mode]),
            "abs_log1p_prediction_delta_vs_matched":
                summary(scale*np.abs(pred[mode]-anchor)),
            "abs_population_weight_delta_vs_matched":
                summary(np.abs(weight[mode]-base_route)),
        }
    baseline=float(np.mean((mean-target)**2))
    for mode in modes:
        report[mode]["relative_mse_to_source_mean"]=(
            report[mode]["mse_log1p"]/max(baseline,1e-12))
    report["source_mean_baseline_mse"]=baseline
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,
                   default=Path("runs/k7/m62_task_query_seed42"))
    p.add_argument("--cache-dir",type=Path,
                   default=Path("runs/k7/m61_structured_seed42"))
    p.add_argument("--output",type=Path,default=None)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    a=p.parse_args()
    if a.batch_size<1:p.error("Positive batch size required")
    if a.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU unavailable")

    config=json.loads((a.run_dir/"config.json").read_text())
    manifest=json.loads((a.cache_dir/"manifest.json").read_text())
    checkpoint=torch.load(a.run_dir/"model.pt",map_location="cpu",
                          weights_only=True)
    if (int(config["seed"])!=42 or tuple(config["source_tasks"])!=TRAIN_TASKS
        or config["heldout_task"]!=HELDOUT_TASK
        or list(checkpoint["group_counts"])!=list(manifest["group_counts"])
        or list(checkpoint["heldout_cities"])!=list(manifest["held_cities"])):
        raise ValueError("Incompatible M6.2 checkpoint/cache/holdout protocol")
    if Path(config["cache_dir"]).resolve()!=a.cache_dir.resolve():
        raise ValueError("M6.2 run was trained with a different cache directory")
    out=a.output or a.run_dir/"frozen_visual_token_audit.json"
    if out.exists():
        raise FileExistsError("Refuse to overwrite existing visual audit: "+str(out))

    frame,x,ctx,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    train,scui,ucui=validate_split(frame,manifest)
    ds=ImageUrbanDataset(manifest["source_csv"],manifest["image_root"],
                         partition="eval")
    if len(ds.df)!=len(frame) or ds.df.indicator_column.astype(str).tolist()!=names:
        raise ValueError("Cached image and source training frame ordering mismatch")
    store=HiddenStore(a.cache_dir/"hidden.sqlite3")
    selected=np.unique(np.concatenate([scui,ucui]))
    hidden=load_precomputed(frame,selected,(HELDOUT_TASK,),ds,store,manifest)
    example=hidden[int(selected[0]),HELDOUT_TASK]
    model=TaskQueriedRouter(
        x.shape[1],example.shape[-1],hidden=int(config["hidden"]),
        heads=int(config["heads"]),group_counts=tuple(manifest["group_counts"]))
    model.load_state_dict(checkpoint["model"],strict=True)
    model=model.to(a.device).eval()
    target_mean,target_std=map(float,checkpoint["target_stats"])
    n_visual=int(manifest["group_counts"][0])
    result={}
    for split,indices in (("SCUI",scui),("UCUI",ucui)):
        result[split]=evaluate_visual_interventions(
            model,x,ctx,y,indices,hidden,n_visual,target_mean,target_std,
            a.batch_size,a.device)
    document={
        "protocol":"Read-only FROZEN M6.2 checkpoint; only first visual-token group intervened. Original RemoteCLIP, context, Definition and Relation tokens held fixed.",
        "source_train_n_not_used":len(train),
        "scui_n":len(scui),"ucui_n":len(ucui),
        "visual_tokens":n_visual,
        "group_counts":manifest["group_counts"],
        "interventions":"matched; shifted within partition visual-only donor; all-zero visual-only. No MLLM recomputation.",
        "warning":"Definition/Relation hidden states may already encode image information. Swapped/zeroed cached visual states are out-of-distribution and not equivalent to coherent joint MLLM counterfactuals; sensitivity alone is not proof of useful visual reasoning.",
        "partitions":result}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(document,indent=2,ensure_ascii=False))
    print(f"[k7-m62-visual] frozen=1 SCUI={len(scui)} UCUI={len(ucui)} "
          f"visual_tokens={n_visual} text_tokens={sum(manifest['group_counts'][1:])}")
    for split,rows in result.items():
        print(f"[k7-m62-visual] {split}")
        for mode in ("matched","swap_visual","zero_visual"):
            row=rows[mode]
            print(f"  {mode:14s} MSE={row['mse_log1p']:.3f} "
                  f"Pearson={row['pearson']:.3f} "
                  f"Ratio={row['relative_mse_to_source_mean']:.3f} "
                  f"|PredDelta|={row['abs_log1p_prediction_delta_vs_matched']['mean']:.5f} "
                  f"|PopWtDelta|={row['abs_population_weight_delta_vs_matched']['mean']:.5f}")
    print(f"[k7-m62-visual] saved {out}")


if __name__=="__main__":
    main()
