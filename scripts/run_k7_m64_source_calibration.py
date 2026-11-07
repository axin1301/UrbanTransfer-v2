"""M6.4 source-only zero-label affine residual calibration.

This is a POST-HOC experiment on the completed M6.3 frozen checkpoint.
Only Pop/Food source training labels fit the affine maps. Source_val,
SCUI, UCSI and UCUI labels are used for reporting AFTER calibration
is frozen; no Education labels enter fitting, moment estimation, tuning
or checkpoint selection. Official test files are untouched.

Per-source normalized correction:
    z_corrected = a_task * z_M63 + b_task
where (a,b) minimize centered least-squares with a fixed conservative
50% shrink toward identity; source-task slopes are clipped to [.75,1.25].
For Education, a,b are combined with frozen Qwen3 source-task semantic
similarity softmax (temperature .2). All constants fixed a priori.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.run_k7_m62_full import CachedTokens,validate_full
from scripts.run_k7_m63_full import PersistentReadOnlyHiddenStore
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.k7_m63_city_indicator_residual import CityIndicatorResidual
from urban_transfer.ktrans_text import qwen_task_vectors

SHRINK=.5
SLOPE_MIN=.75
SLOPE_MAX=1.25
TEMPERATURE=.2


def fit_source_affine(pred,y,names,train_ids,stats):
    """Fit ONLY source train rows in source-standardized coordinates."""
    result={}
    for task in TRAIN_TASKS:
        indices=np.asarray(
            [int(i) for i in train_ids if names[int(i)]==task],dtype=np.int64)
        if len(indices)<2:
            raise ValueError("Not enough source rows: "+task)
        z=np.asarray(pred[indices],dtype=np.float64)
        truth=(np.asarray(y[indices],dtype=np.float64)-stats[task][0])/stats[task][1]
        if not np.isfinite(z).all() or not np.isfinite(truth).all():
            raise ValueError("Nonfinite source values")
        zm=float(z.mean())
        ym=float(truth.mean())
        centered=z-zm
        variance=float(np.dot(centered,centered))
        if variance<1e-10:
            raise ValueError("Degenerate source predictions for "+task)
        ols_slope=float(np.dot(centered,truth-ym)/variance)
        if not np.isfinite(ols_slope):
            raise ValueError("Nonfinite fitted slope")
        # Fixed 50% shrink toward identity. Do not tune on Education
        # or on source_val. Keep slopes positive to preserve Pearson.
        slope=float(np.clip(1.+SHRINK*(ols_slope-1.),
                            SLOPE_MIN,SLOPE_MAX))
        bias=float(SHRINK*(ym-slope*zm))
        result[task]={"slope":slope,"bias":bias,"n_fit":len(indices),
                      "raw_ols_slope":ols_slope,
                      "source_z_mean":zm,"source_y_mean":ym}
    return result


def semantic_weights(embeddings):
    vectors={
        t:F.normalize(embeddings[t].float().flatten(),dim=0)
        for t in (*TRAIN_TASKS,HELDOUT_TASK)}
    similarities=torch.stack([
        F.cosine_similarity(vectors[HELDOUT_TASK],vectors[t],dim=0)
        for t in TRAIN_TASKS])
    weights=torch.softmax(similarities/TEMPERATURE,dim=0).tolist()
    return {t:float(w) for t,w in zip(TRAIN_TASKS,weights)}


def apply_calibration(pred,indices,names,source_affine,weights):
    prediction=np.asarray(pred[indices],dtype=np.float64)
    result=prediction.copy()
    target_a=sum(weights[t]*source_affine[t]["slope"] for t in TRAIN_TASKS)
    target_b=sum(weights[t]*source_affine[t]["bias"] for t in TRAIN_TASKS)
    for task in (*TRAIN_TASKS,HELDOUT_TASK):
        mask=np.asarray([names[int(i)]==task for i in indices],dtype=bool)
        if not mask.any():
            continue
        if task==HELDOUT_TASK:
            a,b=target_a,target_b
        else:
            a,b=source_affine[task]["slope"],source_affine[task]["bias"]
        result[mask]=a*prediction[mask]+b
    return result


@torch.inference_mode()
def predict_normalized(model,x,ctx,indices,names,cache,batch_size,device,stage='inference'):
    outputs=[]
    model.eval()
    start_time=time.monotonic()
    n_batches=(len(indices)+batch_size-1)//batch_size
    print(f'[k7-m64] {stage} START n={len(indices)} batches={n_batches}',flush=True)
    for start in range(0,len(indices),batch_size):
        ix=indices[start:start+batch_size]
        h=cache.batch(ix,[names[int(i)] for i in ix],device)
        outputs.append(model(x[ix].to(device),ctx[ix].to(device),h)
                       .detach().cpu().numpy())
        batch_no=start//batch_size+1
        if batch_no%10==0 or batch_no==n_batches:
            print(f'[k7-m64] {stage} batch={batch_no}/{n_batches} '
                  f'elapsed_s={time.monotonic()-start_time:.1f}',flush=True)
    return np.concatenate(outputs)


def score_pair(partition,mode,indices,pred_normalized,names,y,
               stats,target_stats,source_affine,weights):
    if mode=="calibrated":
        z=apply_calibration(
            pred_normalized,indices,names,source_affine,weights)
    elif mode=="original":
        z=np.asarray(pred_normalized[indices],dtype=np.float64)
    else:
        raise ValueError("Unknown calibration mode")
    means=np.asarray([
        target_stats[0] if names[int(i)]==HELDOUT_TASK
        else stats[names[int(i)]][0] for i in indices])
    scales=np.asarray([
        target_stats[1] if names[int(i)]==HELDOUT_TASK
        else stats[names[int(i)]][1] for i in indices])
    predicted=means+scales*z
    true=np.asarray(y[indices],dtype=np.float64)
    if not (np.isfinite(predicted).all() and np.isfinite(true).all()):
        raise ValueError("Nonfinite M6.4 evaluation data")
    mse=float(np.mean((predicted-true)**2))
    pearson=(float(np.corrcoef(predicted,true)[0,1])
             if min(predicted.std(),true.std())>1e-10 else float("nan"))
    bias=float(predicted.mean()-true.mean())
    return {
        "model":"m64_source_affine" if mode=="calibrated" else "m63_frozen",
        "partition":partition,
        "n_rows":int(len(indices)),
        "mse_log1p":mse,
        "mae_log1p":float(np.mean(np.abs(predicted-true))),
        "pearson":pearson,
        "relative_mse":mse/max(float(np.mean((means-true)**2)),1e-12),
        "prediction_mean":float(predicted.mean()),
        "truth_mean":float(true.mean()),
        "mean_bias_log1p":bias,
        "pred_std":float(predicted.std()),
        "truth_std":float(true.std()),
        "centered_error_mse":float(np.mean(
            ((predicted-predicted.mean())-(true-true.mean()))**2)),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-dir",type=Path,
                        default=Path("runs/k7/m62_full_seed42"))
    parser.add_argument("--m63-dir",type=Path,
                        default=Path("runs/k7/m63_city_indicator_full_seed42_bs256"))
    parser.add_argument("--output",type=Path,
                        default=Path("runs/k7/m64_source_calibration_seed42"))
    parser.add_argument("--text-cache-dir",type=Path,
                        default=Path("runs/ktrans_text"))
    parser.add_argument("--text-model",default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=parser.parse_args()
    if args.batch_size<1:
        parser.error("Positive inference batch size required")
    if args.output.exists():
        raise FileExistsError("M6.4 output already exists; use new directory")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    print("[k7-m64] loading frozen source tensors and M6.3 checkpoint",flush=True)
    manifest_path=(args.full_dir/"full_manifest.json").resolve()
    manifest=json.loads(manifest_path.read_text())
    frame,x,ctx,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    splits=validate_full(frame,manifest)
    m63_path=(args.m63_dir/"m63.pt").resolve()
    checkpoint=torch.load(m63_path,map_location="cpu",weights_only=True)
    if (int(checkpoint["seed"])!=42
        or int(checkpoint["epochs"])!=20
        or Path(checkpoint["full_manifest"]).resolve()!=manifest_path
        or list(checkpoint["heldout_cities"])!=list(manifest["held_cities"])
        or list(checkpoint["group_counts"])!=list(manifest["group_counts"])):
        raise ValueError("M6.3 checkpoint does not match M6.2 full protocol")
    stats={k:(float(v[0]),float(v[1]))
           for k,v in checkpoint["source_stats"].items()}
    if set(stats)!=set(TRAIN_TASKS):
        raise ValueError("Only source Pop/Food statistics permitted")
    target_stats=tuple(map(float,checkpoint["target_stats"]))
    if len(target_stats)!=2 or target_stats[1]<=0:
        raise ValueError("Invalid Education source-imputed moments")
    ds=ImageUrbanDataset(manifest["source_csv"],manifest["image_root"],
                         partition="eval")
    if len(ds.df)!=len(frame) or names!=ds.df.indicator_column.astype(str).tolist():
        raise ValueError("Full-source tensor/token alignment mismatch")
    store=PersistentReadOnlyHiddenStore(args.full_dir/"hidden.sqlite3")
    try:
        cache=CachedTokens(frame,ds,manifest,store,maxsize=512)
        first=cache[int(splits["train"][0]),names[int(splits["train"][0])]]
        model=CityIndicatorResidual(
            x.shape[-1],first.shape[-1],hidden=128,heads=4,
            group_counts=tuple(manifest["group_counts"]),max_residual=.5)
        model.load_state_dict(checkpoint["model"],strict=True)
        model=model.to(args.device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        # Predict source train first. Only source labels are indexed below.
        train_ids=splits["train"]
        train_z=predict_normalized(
            model,x,ctx,train_ids,names,cache,args.batch_size,args.device,
            stage='source_train_fit_inference')
        all_predictions=np.empty(len(frame),dtype=np.float64)
        all_predictions.fill(np.nan)
        all_predictions[train_ids]=train_z
        source_fit=fit_source_affine(
            all_predictions,y.numpy(),names,train_ids,stats)
        print('[k7-m64] source affine coefficients fitted from Pop/Food train only',flush=True)
        embeddings=qwen_task_vectors(
            root=args.text_cache_dir,model_id=args.text_model,
            profile="short",only_cached=True)
        weights=semantic_weights(embeddings)
        target_mapping={
            "slope":float(sum(weights[t]*source_fit[t]["slope"]
                               for t in TRAIN_TASKS)),
            "bias":float(sum(weights[t]*source_fit[t]["bias"]
                              for t in TRAIN_TASKS))}
        # Source fitting is now entirely frozen, before reading any
        # development labels for evaluation.
        results=[]
        partitions=("source_val","SCUI","UCSI","UCUI")
        for split in partitions:
            ix=splits[split]
            all_predictions[ix]=predict_normalized(
                model,x,ctx,ix,names,cache,args.batch_size,args.device,
                stage=split)
            for mode in ("original","calibrated"):
                results.append(score_pair(
                    split,mode,ix,all_predictions,names,y.numpy(),
                    stats,target_stats,source_fit,weights))
            print(f"[k7-m64] scored {split} n={len(ix)}",flush=True)
        for task in TRAIN_TASKS:
            ix=np.asarray([
                int(i) for i in splits["UCSI"]
                if names[int(i)]==task],dtype=np.int64)
            for mode in ("original","calibrated"):
                results.append(score_pair(
                    "UCSI_"+task,mode,ix,all_predictions,names,y.numpy(),
                    stats,target_stats,source_fit,weights))
    finally:
        store.close()
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/"metrics.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (args.output/"calibration.json").write_text(json.dumps({
        "method":"source_only_shrunk_affine_v1",
        "fit_partition":"train",
        "fit_tasks":list(TRAIN_TASKS),
        "source_affine":source_fit,
        "semantic_weights":weights,
        "education_source_only_transfer":target_mapping,
        "slope_shrink":SHRINK,
        "slope_clip":[SLOPE_MIN,SLOPE_MAX],
        "similarity_temperature":TEMPERATURE,
        "m63_checkpoint":str(m63_path),
        "manifest":str(manifest_path),
        "source_val_used_for_fit":False,
        "education_labels_used_for_fit":False,
        "official_test_files_accessed":False,
    },indent=2))
    (args.output/"config.json").write_text(json.dumps({
        k:str(v) if isinstance(v,Path) else v
        for k,v in vars(args).items()},indent=2))
    print("[k7-m64] Education transferred affine",target_mapping,
          "source weights",weights,flush=True)
    for row in results:
        print(f"[k7-m64] {row['model']:20s} "
              f"{row['partition']:28s} MSE={row['mse_log1p']:.4f} "
              f"Pearson={row['pearson']:.4f} bias={row['mean_bias_log1p']:+.4f} "
              f"pred_std={row['pred_std']:.4f}",flush=True)
    print("[k7-m64] COMPLETE",args.output,flush=True)


if __name__=="__main__":
    main()
