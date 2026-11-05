"""Frozen M6.2 calibration decomposition and attention-value comparison.

All fits and MLLM encoders remain frozen. Pseudo-dev Education labels are
used only for descriptive error decomposition. Same-image task differences
use SOURCE training tiles and NO labels. Never opens official test files.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.audit_k7_m62_attention_mechanism import (
    MODES,attention_diagnostics,masked_forward)
from scripts.run_k7_m61_student import load_precomputed
from scripts.run_k7_m62_task_query import validate_split
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.k7_mllm_hidden import HiddenStore,TASKS
from urban_transfer.k7_m62_task_query import TaskQueriedRouter


def summarize(values):
    a=np.asarray(values,dtype=np.float64).reshape(-1)
    if a.size==0 or not np.isfinite(a).all():
        raise ValueError("Empty/nonfinite calibration-audit values")
    return {"mean":float(a.mean()),"std":float(a.std()),
            "min":float(a.min()),"max":float(a.max())}


def mse_decomposition(prediction,truth):
    """Exact population-moment identity with ddof=0.

    MSE = (mean(p)-mean(y))² + (std(p)-std(y))²
          + 2*std(p)*std(y)*(1-Pearson(p,y)).
    The last term is positive and computed by covariance for stability
    even when a constant prediction makes Pearson undefined.
    """
    p=np.asarray(prediction,dtype=np.float64).reshape(-1)
    y=np.asarray(truth,dtype=np.float64).reshape(-1)
    if len(p)<2 or p.shape!=y.shape or not (
        np.isfinite(p).all() and np.isfinite(y).all()):
        raise ValueError("Invalid predictions or truth")
    mp,my=float(p.mean()),float(y.mean())
    sp,sy=float(p.std()),float(y.std())
    cov=float(np.mean((p-mp)*(y-my)))
    bias=(mp-my)**2
    scale=(sp-sy)**2
    correlation=max(0.,2*(sp*sy-cov))
    mse=float(np.mean((p-y)**2))
    reconstructed=bias+scale+correlation
    if not np.isclose(mse,reconstructed,rtol=1e-9,atol=1e-9):
        raise RuntimeError("Error components do not reproduce MSE")
    pearson=(cov/(sp*sy) if min(sp,sy)>1e-10 else None)
    return {
        "mse":mse,"mean_prediction":mp,"mean_truth":my,
        "std_prediction_ddof0":sp,"std_truth_ddof0":sy,
        "mean_signed_error":mp-my,
        "pearson":pearson,
        "mean_bias_squared":bias,
        "std_mismatch_squared":scale,
        "correlation_component":correlation,
        "mse_reconstructed":reconstructed,
    }


@torch.inference_mode()
def dev_calibration(model,x,ctx,y,indices,cache,
                    mean,scale,batch_size,device):
    predictions={mode:[] for mode in MODES}
    for start in range(0,len(indices),batch_size):
        ix=indices[start:start+batch_size]
        tok=torch.stack([cache[int(i),HELDOUT_TASK] for i in ix]).to(device)
        q,att,fused,_=attention_diagnostics(model,tok)
        xv=x[ix].to(device)
        cv=ctx[ix].to(device)
        for mode in MODES:
            result,_=masked_forward(model,xv,cv,q,fused,att,mode)
            predictions[mode].append(result.cpu().numpy())
        # Verify the native trained forward; diagnosis must not drift.
        native=model(xv,cv,tok).cpu().numpy()
        if not np.allclose(native,predictions["normal"][-1],
                           atol=1e-5,rtol=1e-5):
            raise RuntimeError("Normal audit does not reconstruct M6.2")
    truth=y[torch.as_tensor(indices,dtype=torch.long)].numpy()
    return {mode:mse_decomposition(
        mean+scale*np.concatenate(values),truth)
        for mode,values in predictions.items()}


def representation_pair(a,b):
    """Paired same-image comparisons in the SAME feature space."""
    if a.shape!=b.shape or a.ndim!=2:
        raise ValueError("Expected aligned [samples, feature_dim]")
    a=a.float()
    b=b.float()
    cosine=F.cosine_similarity(a,b,dim=-1)
    delta=(a-b).norm(dim=-1)
    size=.5*(a.norm(dim=-1)+b.norm(dim=-1))
    return {"cosine":summarize(cosine.numpy()),
            "absolute_l2":summarize(delta.numpy()),
            "relative_l2":summarize((delta/size.clamp_min(1e-8)).numpy())}


@torch.inference_mode()
def same_image_stage_differences(model,indices,cache,batch_size,device):
    """Source tiles only, three indicator prompts on each identical tile."""
    per_task={task:{"query":[],"attention_output":[],"fusion_output":[]}
              for task in TASKS}
    for start in range(0,len(indices),batch_size):
        ix=indices[start:start+batch_size]
        for task in TASKS:
            tokens=torch.stack([cache[int(i),task] for i in ix]).to(device)
            q,att,fused,_=attention_diagnostics(model,tokens)
            per_task[task]["query"].append(q.cpu())
            per_task[task]["attention_output"].append(att.cpu())
            per_task[task]["fusion_output"].append(fused.cpu())
    vectors={task:{stage:torch.cat(chunks,dim=0)
                   for stage,chunks in stages.items()}
             for task,stages in per_task.items()}
    # A comparison of outputs also checks whether changing attention
    # weights leads to different attention-weighted VISUAL VALUE vectors.
    results={}
    for left,right in (
        (TRAIN_TASKS[0],HELDOUT_TASK),
        (TRAIN_TASKS[1],HELDOUT_TASK),
        (TRAIN_TASKS[0],TRAIN_TASKS[1])):
        key=left+"__vs__"+right
        results[key]={
            stage:representation_pair(
                vectors[left][stage],vectors[right][stage])
            for stage in ("query","attention_output","fusion_output")}
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,
                        default=Path("runs/k7/m62_task_query_seed42"))
    parser.add_argument("--cache-dir",type=Path,
                        default=Path("runs/k7/m61_structured_seed42"))
    parser.add_argument("--output",type=Path,default=None)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    a=parser.parse_args()
    if a.batch_size<1:parser.error("Positive batch-size required")
    if a.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm requested but unavailable")
    output=a.output or a.run_dir/"frozen_calibration_attention_audit.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite previous audit: "+str(output))
    config=json.loads((a.run_dir/"config.json").read_text())
    manifest=json.loads((a.cache_dir/"manifest.json").read_text())
    checkpoint=torch.load(a.run_dir/"model.pt",map_location="cpu",weights_only=True)
    if (int(config["seed"])!=42 or tuple(config["source_tasks"])!=TRAIN_TASKS
        or config["heldout_task"]!=HELDOUT_TASK
        or Path(config["cache_dir"]).resolve()!=a.cache_dir.resolve()
        or list(checkpoint["group_counts"])!=list(manifest["group_counts"])
        or list(checkpoint["heldout_cities"])!=list(manifest["held_cities"])):
        raise ValueError("M6.2 checkpoint/cache/protocol disagreement")
    frame,x,ctx,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    train,scui,ucui=validate_split(frame,manifest)
    ds=ImageUrbanDataset(manifest["source_csv"],manifest["image_root"],
                         partition="eval")
    if len(ds.df)!=len(frame) or names!=ds.df.indicator_column.astype(str).tolist():
        raise ValueError("Source image/cache row order differs")
    store=HiddenStore(a.cache_dir/"hidden.sqlite3")
    source_cache=load_precomputed(frame,train,TASKS,ds,store,manifest)
    eval_ids=np.unique(np.concatenate((scui,ucui)))
    eval_cache=load_precomputed(frame,eval_ids,(HELDOUT_TASK,),ds,store,manifest)
    sample=source_cache[int(train[0]),HELDOUT_TASK]
    model=TaskQueriedRouter(
        x.shape[-1],sample.shape[-1],hidden=int(config["hidden"]),
        heads=int(config["heads"]),group_counts=tuple(manifest["group_counts"]))
    model.load_state_dict(checkpoint["model"],strict=True)
    model=model.to(a.device).eval()
    mean,scale=map(float,checkpoint["target_stats"])
    if scale<=0:raise ValueError("Invalid target calibration scale")
    dev={
        split:dev_calibration(model,x,ctx,y,ids,eval_cache,
                              mean,scale,a.batch_size,a.device)
        for split,ids in (("SCUI",scui),("UCUI",ucui))}
    source=same_image_stage_differences(
        model,train,source_cache,a.batch_size,a.device)
    report={
        "protocol":"Frozen M6.2, target labels SCUI/UCUI only for post-fit reporting; same-image source stage comparisons never use labels; no official test",
        "source_rows":len(train),"scui_rows":len(scui),"ucui_rows":len(ucui),
        "population_variance_ddof":0,
        "mse_identity":"(mean_pred-mean_true)^2 + (std_pred-std_true)^2 + 2*(std_pred*std_true-covariance)",
        "dev_calibration":dev,
        "same_image_source_stages":source,
        "caveat":"Off-manifold zero-ablation improvements cannot select model or calibration on Education pseudo-dev. Attention output cosine != proof of semantic visual reasoning.",
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(f"[k7-m62-cal] frozen=1 source={len(train)} SCUI={len(scui)} UCUI={len(ucui)}")
    for split,records in dev.items():
        print(f"[k7-m62-cal] {split}")
        for mode in MODES:
            row=records[mode]
            corr="nan" if row["pearson"] is None else f"{row['pearson']:.3f}"
            print(f"  {mode:22s} MSE={row['mse']:.3f} Pearson={corr} "
                  f"PredMean={row['mean_prediction']:.3f} PredStd={row['std_prediction_ddof0']:.3f} "
                  f"TruthMean={row['mean_truth']:.3f} TruthStd={row['std_truth_ddof0']:.3f} "
                  f"Bias2={row['mean_bias_squared']:.3f} "
                  f"Std2={row['std_mismatch_squared']:.3f} "
                  f"CorrTerm={row['correlation_component']:.3f}")
    print("[k7-m62-cal] SOURCE same-image indicator stage comparisons:")
    for pair,parts in source.items():
        print("  "+pair)
        for stage in ("query","attention_output","fusion_output"):
            row=parts[stage]
            print(f"    {stage:17s} cosine={row['cosine']['mean']:.5f} "
                  f"relative_L2={row['relative_l2']['mean']:.5f}")
    print(f"[k7-m62-cal] saved {output}")


if __name__=="__main__":
    main()
