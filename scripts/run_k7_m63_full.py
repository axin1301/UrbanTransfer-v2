"""Train M6.3 on FULL source Pop/Food data against frozen visual anchor.

No Qwen encoding: reuse M6.2 full SQLite and exact full_manifest.json.
No official test files. Target Education labels are only read for
post-fit descriptive SCUI/UCUI reporting, never optimizer or calibration.
A single fixed 20-epoch run (seed42), no target-driven model selection.
"""
import argparse
import csv
import json
import os
import time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.run_k7_m62_full import (
    CachedTokens,validate_full,source_train_targets,evaluate)
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.k7_mllm_hidden import HiddenStore
from urban_transfer.k7_m63_city_indicator_residual import CityIndicatorResidual


def read_reference(reference_dir,manifest_path,manifest):
    checkpoint=torch.load(reference_dir/"visual_only.pt",
                          map_location="cpu",weights_only=True)
    if (checkpoint.get("variant")!="visual_only"
        or int(checkpoint["seed"])!=42
        or list(checkpoint["heldout_cities"])!=list(manifest["held_cities"])
        or Path(checkpoint["manifest_path"]).resolve()!=manifest_path.resolve()):
        raise ValueError("Frozen Visual-only anchor is not from this full manifest")
    original_config=json.loads((reference_dir/"config.json").read_text())
    if (int(original_config["seed"])!=42 or
        int(original_config["hidden"])!=128):
        raise ValueError("Frozen visual baseline architecture mismatch")
    stats={name:(float(v[0]),float(v[1]))
           for name,v in checkpoint["source_stats"].items()}
    target=tuple(map(float,checkpoint["unseen_stats"]))
    if set(stats)!=set(TRAIN_TASKS) or any(s[1]<=0 for s in stats.values()):
        raise ValueError("Expected original source-only Pop/Food standardization")
    if len(target)!=2 or target[1]<=0:
        raise ValueError("Invalid frozen Education source-imputed moments")
    return checkpoint,stats,target


@torch.inference_mode()
def evaluate_m63(model,split,indices,x,ctx,y,names,stats,
                 target_moments,cache,batch_size,device):
    model.eval()
    predictions=[]
    weights=[]
    bases=[]
    transfers=[]
    for start in range(0,len(indices),batch_size):
        ids=indices[start:start+batch_size]
        visual=x[ids].to(device)
        geo=ctx[ids].to(device)
        h=cache.batch(ids,[names[int(i)] for i in ids],device)
        pred,w,_,_,base,delta=model(
            visual,geo,h,return_aux=True)
        predictions.append(pred.cpu().numpy())
        weights.append(w[:,0].cpu().numpy())
        bases.append(base.cpu().numpy())
        transfers.append(delta.cpu().numpy())
    normalized=np.concatenate(predictions)
    means=np.asarray([
        target_moments[0] if names[int(i)]==HELDOUT_TASK
        else stats[names[int(i)]][0] for i in indices],dtype=np.float64)
    scales=np.asarray([
        target_moments[1] if names[int(i)]==HELDOUT_TASK
        else stats[names[int(i)]][1] for i in indices],dtype=np.float64)
    prediction=means+scales*normalized
    truth=y[torch.as_tensor(indices,dtype=torch.long)].numpy()
    mse=float(np.mean((prediction-truth)**2))
    base_mse=float(np.mean((means-truth)**2))
    pstd=float(prediction.std())
    ystd=float(truth.std())
    pearson=(float(np.corrcoef(prediction,truth)[0,1])
             if min(pstd,ystd)>1e-10 else float("nan"))
    route=np.concatenate(weights)
    delta=np.concatenate(transfers)
    original=np.concatenate(bases)
    return {
        "model":"m63","partition":split,
        "n_rows":int(len(indices)),
        "mse_log1p":mse,
        "mae_log1p":float(np.mean(np.abs(prediction-truth))),
        "pearson":pearson,
        "source_mean_mse":base_mse,
        "relative_mse":mse/max(base_mse,1e-12),
        "population_weight":float(route.mean()),
        "population_weight_std":float(route.std()),
        "prediction_mean":float(prediction.mean()),
        "truth_mean":float(truth.mean()),
        "mean_abs_transfer_normalized":float(np.mean(np.abs(delta))),
        "max_abs_transfer_normalized":float(np.max(np.abs(delta))),
        "mean_abs_transfer_log1p":float(np.mean(np.abs(delta)*scales)),
        "base_mean_normalized":float(original.mean()),
    }


def train_m63(model,x,ctx,y,names,train_ids,stats,cache,args,device,checkpoint_path=None,signature=None):
    target=source_train_targets(train_ids,names,y,stats)
    source_labels=torch.tensor(
        [TRAIN_TASKS.index(names[int(i)]) for i in train_ids],
        dtype=torch.long)
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,weight_decay=1e-4)
    order_rng=torch.Generator().manual_seed(args.seed)
    start_epoch=0
    if checkpoint_path is not None and checkpoint_path.exists():
        saved=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
        if saved.get("signature")!=signature or saved.get("version")!=1:
            raise ValueError("Resume checkpoint settings/manifest/anchor mismatch")
        start_epoch=int(saved["completed_epoch"])
        if start_epoch<1 or start_epoch>args.epochs:
            raise ValueError("Invalid completed_epoch in resume checkpoint")
        model.load_state_dict(saved["model"],strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        order_rng.set_state(saved["order_rng"])
        torch.set_rng_state(saved["torch_cpu_rng"])
        if device.type=="cuda":
            if "torch_cuda_rng" not in saved:
                raise ValueError("CUDA RNG state missing from checkpoint")
            torch.cuda.set_rng_state(saved["torch_cuda_rng"],device)
        print(f"[k7-m63] RESUME completed_epoch={start_epoch} "
              f"remaining={args.epochs-start_epoch}",flush=True)
    for epoch in range(start_epoch,args.epochs):
        epoch_started=time.monotonic()
        print(f"[k7-m63] epoch={epoch+1}/{args.epochs} START",flush=True)
        model.train()
        # Base has no dropout/batchnorm, but must remain in eval mode.
        model.base.eval()
        order=torch.randperm(len(train_ids),generator=order_rng)
        losses=[]
        n_batches=(len(train_ids)+args.batch_size-1)//args.batch_size
        for start in range(0,len(train_ids),args.batch_size):
            batch=order[start:start+args.batch_size]
            ix=train_ids[batch.numpy()]
            v=x[ix].to(device)
            context=ctx[ix].to(device)
            yy=target[batch].to(device)
            source_idx=source_labels[batch].to(device)
            source_h=cache.batch(ix,[names[int(i)] for i in ix],device)
            pop_h=cache.batch(ix,[TRAIN_TASKS[0]]*len(ix),device)
            food_h=cache.batch(ix,[TRAIN_TASKS[1]]*len(ix),device)
            edu_h=cache.batch(ix,[HELDOUT_TASK]*len(ix),device)
            optimizer.zero_grad(set_to_none=True)
            prediction,w,task,fused,baseline,delta=model(
                v,context,source_h,return_aux=True)
            own=model.source_expert_predictions(
                v,context,source_h).gather(
                    1,source_idx[:,None]).squeeze(1)
            task_cls=model.source_task_separation(
                v,context,pop_h,food_h)
            align=model.target_image_alignment(v,context,edu_h)
            loss=(F.mse_loss(prediction,yy)+
                  args.self_loss*F.mse_loss(own,yy)+
                  args.task_loss*task_cls+
                  args.target_align_loss*align+
                  args.residual_penalty*delta.square().mean())
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite M6.3 source training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],2.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            batch_number=start//args.batch_size+1
            if batch_number%50==0 or batch_number==n_batches:
                print(f"[k7-m63] epoch={epoch+1}/{args.epochs} "
                      f"batch={batch_number}/{n_batches} "
                      f"elapsed_s={time.monotonic()-epoch_started:.1f}",
                      flush=True)
        if checkpoint_path is not None:
            progress={
                "version":1,
                "signature":signature,
                "completed_epoch":epoch+1,
                "model":{k:v.detach().cpu().clone()
                         for k,v in model.state_dict().items()},
                "optimizer":optimizer.state_dict(),
                "order_rng":order_rng.get_state(),
                "torch_cpu_rng":torch.get_rng_state(),
            }
            if device.type=="cuda":
                progress["torch_cuda_rng"]=torch.cuda.get_rng_state(device)
            temporary=checkpoint_path.with_name(
                checkpoint_path.name+".tmp")
            torch.save(progress,temporary)
            os.replace(temporary,checkpoint_path)
        print(f"[k7-m63] epoch={epoch+1}/{args.epochs} "
              f"source_loss={np.mean(losses):.5f} "
              f"elapsed_s={time.monotonic()-epoch_started:.1f} "
              f"checkpoint_saved={checkpoint_path is not None}",flush=True)
    model.eval()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-dir",type=Path,
                   default=Path("runs/k7/m62_full_seed42"))
    p.add_argument("--reference-dir",type=Path,
                   default=Path("runs/k7/m62_full_models_seed42"))
    p.add_argument("--output",type=Path,
                   default=Path("runs/k7/m63_city_indicator_full_seed42"))
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--hidden",type=int,default=128)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--self-loss",type=float,default=.3)
    p.add_argument("--task-loss",type=float,default=.2)
    p.add_argument("--target-align-loss",type=float,default=.05)
    p.add_argument("--residual-penalty",type=float,default=.02)
    p.add_argument("--max-residual",type=float,default=.5)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=p.parse_args()
    if (args.seed!=42 or args.epochs!=20 or args.batch_size!=128
        or args.hidden!=128 or args.heads!=4 or args.lr!=.001
        or args.self_loss!=.3 or args.task_loss!=.2
        or args.target_align_loss!=.05
        or args.residual_penalty!=.02 or args.max_residual!=.5):
        p.error("Fixed first M6.3 experiment settings required")
    if args.output.resolve() in {
            args.full_dir.resolve(),args.reference_dir.resolve()}:
        raise ValueError("M6.3 output must differ from caches and reference")
    checkpoint_path=args.output/"checkpoint_last.pt"
    if args.output.exists() and not checkpoint_path.is_file():
        raise FileExistsError(
            "M6.3 output exists without resumable checkpoint; "
            "use a distinct --output path")
    if (args.output/"metrics.csv").exists():
        raise FileExistsError("M6.3 run already completed; refusing retrain")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("ROCm MI250 GPU unavailable")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    manifest_path=args.full_dir/"full_manifest.json"
    manifest=json.loads(manifest_path.read_text())
    frame,x,ctx,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    splits=validate_full(frame,manifest)
    ref,stats,target_moments=read_reference(
        args.reference_dir,manifest_path,manifest)
    train_ids=splits["train"]
    if set(frame.iloc[train_ids].indicator_column)!=set(TRAIN_TASKS):
        raise ValueError("Full M6.3 must train both source indicators")
    ds=ImageUrbanDataset(manifest["source_csv"],manifest["image_root"],
                         partition="eval")
    if (len(ds.df)!=len(frame) or
        names!=ds.df.indicator_column.astype(str).tolist()):
        raise ValueError("Shared visual source cache ordering mismatch")
    token_cache=CachedTokens(
        frame,ds,manifest,HiddenStore(args.full_dir/"hidden.sqlite3"))
    sample=token_cache[int(train_ids[0]),names[int(train_ids[0])]]
    device=torch.device(args.device)
    model=CityIndicatorResidual(
        x.shape[-1],sample.shape[-1],hidden=args.hidden,
        heads=args.heads,group_counts=tuple(manifest["group_counts"]),
        max_residual=args.max_residual).to(device)
    model.load_visual_baseline(ref["model"])
    # Critical invariant: before M6.3 optimization, predictions
    # EXACTLY equal the frozen Visual-only anchor.
    probe=train_ids[:min(len(train_ids),16)]
    with torch.inference_mode():
        h=token_cache.batch(probe,[names[int(i)] for i in probe],device)
        v=x[probe].to(device)
        c=ctx[probe].to(device)
        initial,_,_,_,base,delta=model(
            v,c,h,return_aux=True)
        if (not torch.allclose(initial,base,atol=1e-7,rtol=1e-7)
            or not torch.allclose(delta,torch.zeros_like(delta),atol=1e-7)):
            raise RuntimeError("M6.3 is not zero-residual initially")
    print(f"[k7-m63] FULL source={len(train_ids)} "
          f"source_val={len(splits['source_val'])} "
          f"SCUI={len(splits['SCUI'])} UCSI={len(splits['UCSI'])} "
          f"UCUI={len(splits['UCUI'])} "
          f"frozen_visual_anchor=1 max_residual={args.max_residual}",
          flush=True)
    signature={
        "manifest":str(manifest_path.resolve()),
        "anchor":str((args.reference_dir/"visual_only.pt").resolve()),
        **{key:getattr(args,key) for key in (
            "seed","epochs","batch_size","hidden","heads","lr",
            "self_loss","task_loss","target_align_loss",
            "residual_penalty","max_residual")}
    }
    args.output.mkdir(parents=True,exist_ok=True)
    train_m63(model,x,ctx,y,names,train_ids,stats,
              token_cache,args,device,checkpoint_path,signature)
    # Label arrays for Education are consulted ONLY AFTER training ends.
    metrics=[]
    for split in ("source_val","SCUI","UCSI","UCUI"):
        metrics.append(evaluate_m63(
            model,split,splits[split],x,ctx,y,names,stats,
            target_moments,token_cache,args.batch_size,device))
    for task in TRAIN_TASKS:
        indices=np.asarray([
            int(i) for i in splits["UCSI"]
            if names[int(i)]==task],dtype=np.int64)
        metrics.append(evaluate_m63(
            model,"UCSI_"+task,indices,x,ctx,y,names,stats,
            target_moments,token_cache,args.batch_size,device))
    # Explicitly compare frozen-baseline metrics to the original output.
    with (args.reference_dir/"metrics.csv").open(newline="") as inp:
        reference={row["partition"]:row
                   for row in csv.DictReader(inp)
                   if row["model"]=="visual_only"}
    torch.save({
        "model":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "seed":args.seed,"epochs":args.epochs,
        "frozen_anchor":str((args.reference_dir/"visual_only.pt").resolve()),
        "full_manifest":str(manifest_path.resolve()),
        "source_stats":stats,"target_stats":target_moments,
        "heldout_cities":manifest["held_cities"],
        "group_counts":manifest["group_counts"]},
        args.output/"m63.pt")
    with (args.output/"metrics.csv").open("w",newline="") as output:
        writer=csv.DictWriter(output,fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (args.output/"config.json").write_text(json.dumps({
        **{k:str(v) if isinstance(v,Path) else v
           for k,v in vars(args).items()},
        "n_source":len(train_ids),
        "n_source_val":len(splits["source_val"]),
        "n_scui":len(splits["SCUI"]),
        "n_ucsi":len(splits["UCSI"]),
        "n_ucui":len(splits["UCUI"]),
        "protocol":"M6.3 pretrained Visual-only frozen source anchor, source-only supervised bounded multimodal transfer, no Education label for fit or selection, no official test."},
        indent=2))
    for row in metrics:
        previous=reference.get(row["partition"])
        comparison=("baseline_MSE="+format(
            float(previous["mse_log1p"]),".4f") if previous else "")
        print(f"[k7-m63] {row['partition']:28s} "
              f"n={row['n_rows']} MSE={row['mse_log1p']:.4f} "
              f"Pearson={row['pearson']:.4f} "
              f"Ratio={row['relative_mse']:.4f} "
              f"TransferAbs={row['mean_abs_transfer_log1p']:.4f} "
              f"{comparison}",flush=True)
    print(f"[k7-m63] COMPLETE {args.output}",flush=True)


if __name__=="__main__":
    main()
