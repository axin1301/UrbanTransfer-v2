"""M6.7 three-arm paired source-city DEV experiment, no target labels.

Each source-train city receives cross-fitted source-indicator predictions
from models that have NOT seen that city. Held cities receive predictions
from models fitted only on the source-fit cities. Auxiliary feature
encoders never consume source-city held labels or official test files.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F

from scripts.run_k7_m62_full import (
    CachedTokens,validate_full,source_train_targets)
from scripts.run_k7_m63_full import (
    PersistentReadOnlyHiddenStore,load_training_tokens)
from scripts.run_k7_m65_dev_intervention import validate_dev_provenance
from scripts.run_k7_m65_robustness_suite import write_csv
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS
from urban_transfer.k7_m65_neighborhood import load_or_build_neighborhood
from urban_transfer.k7_m66_indicator_graph import cached_physical_neighbor_graph
from urban_transfer.k7_m67_oof_indicator_context import load_crossfit_context
from urban_transfer.k7_m67_predictive_neighbor import PredictiveNeighborResidual

ARMS=("cross_indicator","same_indicator","no_predictive")


def arm_predicted_features(source_preds,graph,ids,names,arm):
    """All arms use same 2 channels/architecture and target A identity."""
    if arm not in ARMS:
        raise ValueError("Unrecognized source-only ablation arm")
    node,dist,valid=graph
    ids=np.asarray(ids,dtype=np.int64)
    index=node[ids]
    predictions=source_preds[index].clone()
    a=np.asarray([TRAIN_TASKS.index(names[int(i)]) for i in ids],
                 dtype=np.int64)
    first=predictions.gather(
        2,torch.from_numpy(a).reshape(-1,1,1).expand(-1,index.shape[1],1))
    second=predictions.gather(
        2,torch.from_numpy(1-a).reshape(-1,1,1).expand(-1,index.shape[1],1))
    if arm=="cross_indicator":
        values=torch.cat((first,second),dim=-1)
    elif arm=="same_indicator":
        values=torch.cat((first,first),dim=-1)
    else:
        values=torch.zeros_like(predictions)
    return values,dist[ids],valid[ids]


def batch_features(ids,visual,geo,neighborhood,graph,prediction,names,arm,device):
    values,dist,valid=arm_predicted_features(
        prediction,graph,ids,names,arm)
    if not bool(valid.any(dim=1).all()):
        raise ValueError("Source tile with no physical same-city neighbor")
    return (visual[ids].to(device),geo[ids].to(device),
            neighborhood[ids].to(device),values.to(device),
            dist.to(device),valid.to(device))


def train_arm(model,arm,visual,geo,neighborhood,graph,prediction,
              y,names,ids,stats,cache,output,signature,args,device):
    checkpoint=output/"checkpoint_last.pt"
    target=source_train_targets(ids,names,y,stats)
    labels=torch.tensor([TRAIN_TASKS.index(names[int(i)]) for i in ids],
                         dtype=torch.long)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=1e-4)
    rng=torch.Generator().manual_seed(42)
    begin=0
    if checkpoint.is_file():
        saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if saved.get("version")!=1 or saved.get("signature")!=signature:
            raise ValueError("M6.7 checkpoint protocol mismatch")
        model.load_state_dict(saved["model"],strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["shuffle_rng"])
        torch.set_rng_state(saved["cpu_rng"])
        if device.type=="cuda":
            torch.cuda.set_rng_state(saved["gpu_rng"],device=device)
        begin=int(saved["completed_epoch"])
        if not 1<=begin<=20:
            raise ValueError("Invalid M6.7 checkpoint epoch")
        print(f"[m67] RESUME {arm} completed_epoch={begin}",flush=True)
    for epoch in range(begin,20):
        started=time.monotonic()
        model.train()
        model.base.eval()
        order=torch.randperm(len(ids),generator=rng)
        losses=[]
        for start in range(0,len(order),args.batch_size):
            local=order[start:start+args.batch_size]
            chosen=ids[local.numpy()]
            v,g,nb,npred,nd,nm=batch_features(
                chosen,visual,geo,neighborhood,graph,prediction,
                names,arm,device)
            source_h,pop_h,food_h,edu_h=load_training_tokens(
                cache,chosen,names,device)
            yy=target[local].to(device)
            label=labels[local].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred,_,_,_,_,delta=model(
                v,g,source_h,nb,npred,nd,nm,return_aux=True)
            own=model.source_expert_predictions(
                v,g,source_h,nb,npred,nd,nm).gather(
                    1,label[:,None]).squeeze(1)
            sep=model.source_task_separation(
                v,g,pop_h,food_h,nb,npred,nd,nm)
            align=model.target_image_alignment(
                v,g,edu_h,nb,npred,nd,nm)
            loss=(F.mse_loss(pred,yy)+.3*F.mse_loss(own,yy)+
                  .2*sep+.05*align+.02*delta.square().mean())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite M6.7 source loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,2.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if (start//args.batch_size+1)%50==0:
                print(f"[m67] {arm} epoch={epoch+1}/20 "
                      f"batch={start//args.batch_size+1} "
                      f"elapsed_s={time.monotonic()-started:.1f}",flush=True)
        state={
            "version":1,"signature":signature,"completed_epoch":epoch+1,
            "model":{k:v.detach().cpu().clone()
                     for k,v in model.state_dict().items()},
            "optimizer":optimizer.state_dict(),
            "shuffle_rng":rng.get_state(),
            "cpu_rng":torch.get_rng_state(),
        }
        if device.type=="cuda":
            state["gpu_rng"]=torch.cuda.get_rng_state(device=device)
        temp=checkpoint.with_name(checkpoint.name+".tmp")
        torch.save(state,temp)
        os.replace(temp,checkpoint)
        print(f"[m67] {arm} epoch={epoch+1}/20 "
              f"source_loss={np.mean(losses):.3f} "
              f"elapsed_s={time.monotonic()-started:.1f} "
              "checkpoint_saved=True",flush=True)


@torch.inference_mode()
def evaluate(model,arm,task,ids,visual,geo,neighborhood,graph,prediction,
             y,names,stats,cache,batch_size,device):
    model.eval()
    values=[]; routes=[]
    for start in range(0,len(ids),batch_size):
        chosen=ids[start:start+batch_size]
        v,g,nb,npred,nd,nm=batch_features(
            chosen,visual,geo,neighborhood,graph,prediction,
            names,arm,device)
        h=cache.batch(chosen,[names[int(i)] for i in chosen],device)
        z,w,*_=model(v,g,h,nb,npred,nd,nm,return_aux=True)
        values.extend(z.detach().cpu().numpy().tolist())
        routes.extend(w[:,0].detach().cpu().numpy().tolist())
    pred=(stats[task][0]+stats[task][1]*
          np.asarray(values,dtype=np.float64))
    truth=y[torch.as_tensor(ids)].numpy().astype(np.float64)
    error=pred-truth
    return {
        "task":task,"arm":arm,"n_rows":len(ids),
        "mse_log1p":float(np.mean(error**2)),
        "mae_log1p":float(np.mean(np.abs(error))),
        "pearson":float(np.corrcoef(pred,truth)[0,1])
            if min(pred.std(),truth.std())>1e-10 else float("nan"),
        "bias":float(error.mean()),"pred_std":float(pred.std()),
        "population_weight":float(np.mean(routes))
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-dir",type=Path,
                   default=Path("runs/k7/m62_full_seed42"))
    p.add_argument("--dev-dir",type=Path,
                   default=Path("runs/k7/m65_neighborhood_dev_seed42"))
    p.add_argument("--neighbor-cache",type=Path,
                   default=Path("runs/k7/m65_neighbors_k8.pt"))
    p.add_argument("--graph-cache",type=Path,
                   default=Path("runs/k7/m66_physical_graph_k8.pt"))
    p.add_argument("--oof-cache",type=Path,
                   default=Path("runs/k7/m67_oof_source_city.pt"))
    p.add_argument("--output",type=Path,
                   default=Path("runs/k7/m67_predictive_cross_dev_seed42"))
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    a=p.parse_args()
    if a.epochs!=20 or a.batch_size!=256:
        p.error("M6.7 requires 20 epochs/batch256 source-only paired protocol")
    if a.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    if a.output.resolve()==a.dev_dir.resolve():
        raise ValueError("Do not overwrite original M6.5 Dev")
    if (a.output/"SUMMARY.csv").exists():
        print(f"[m67] ALREADY COMPLETE {a.output}",flush=True)
        return
    manifest_path=(a.full_dir/"full_manifest.json").resolve()
    manifest=json.loads(manifest_path.read_text())
    frame,visual,geo,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    splits=validate_full(frame,manifest)
    _,train_ids,held_ids,cities,stats=validate_dev_provenance(
        a.dev_dir,manifest_path,frame,splits,y,names)
    neighborhood=load_or_build_neighborhood(
        frame,visual,manifest_path,a.neighbor_cache,k=8)
    graph=cached_physical_neighbor_graph(
        frame,manifest_path,a.graph_cache,k=8)
    oof=load_crossfit_context(
        frame,visual,geo,y,names,train_ids,stats,manifest_path,a.oof_cache)
    anchor_path=a.dev_dir/"dev_visual_anchor.pt"
    anchor=torch.load(anchor_path,map_location="cpu",weights_only=True)
    config={
        "protocol":"m67_source_city_oof_cross_predictive_v1",
        "manifest":str(manifest_path),
        "source_sha256":hashlib.sha256(
            np.asarray(train_ids,dtype=np.int64).tobytes()).hexdigest(),
        "held_sha256":hashlib.sha256(
            np.asarray(held_ids,dtype=np.int64).tobytes()).hexdigest(),
        "anchor_sha256":hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        "oof_cache":str(a.oof_cache.resolve()),
        "graph_cache":str(a.graph_cache.resolve()),
        "neighbor_cache":str(a.neighbor_cache.resolve()),
        "arms":list(ARMS),"epochs":20,"batch_size":256,
        "initial_seed":42,"shuffle_seed":42,"cities":cities,
        "oof_folds":4,"ridge_alpha":100.,
        "heldout_task_labels_read":False,"official_test_labels_read":False,
        "crossfit_source_cities_excluded":True,
    }
    a.output.mkdir(parents=True,exist_ok=True)
    cfg=a.output/"config.json"
    if cfg.exists():
        if json.loads(cfg.read_text())!=config:
            raise ValueError("Existing M6.7 run config mismatch")
    else:
        if any(a.output.iterdir()):
            raise FileExistsError("Unrecognized M6.7 output folder")
        tmp=cfg.with_suffix(".tmp")
        tmp.write_text(json.dumps(config,indent=2))
        os.replace(tmp,cfg)
    ds=ImageUrbanDataset(manifest["source_csv"],manifest["image_root"],
                         partition="eval")
    if len(ds.df)!=len(frame) or names!=ds.df.indicator_column.astype(str).tolist():
        raise ValueError("Source image / Qwen token rows misaligned")
    store=PersistentReadOnlyHiddenStore(a.full_dir/"hidden.sqlite3")
    try:
        cache=CachedTokens(frame,ds,manifest,store,maxsize=512)
        first=cache[int(train_ids[0]),names[int(train_ids[0])]]
        device=torch.device(a.device)
        torch.manual_seed(42)
        if device.type=="cuda":
            torch.cuda.manual_seed_all(42)
        template=PredictiveNeighborResidual(
            visual.shape[-1],first.shape[-1],hidden=128,heads=4,
            group_counts=tuple(manifest["group_counts"]))
        template.load_visual_baseline(anchor["model"])
        initial={k:v.detach().cpu().clone()
                 for k,v in template.state_dict().items()}
        cpu_rng=torch.get_rng_state().clone()
        gpu_rng=(torch.cuda.get_rng_state(device=device).clone()
                 if device.type=="cuda" else None)
        del template
        all_rows=[]
        for arm in ARMS:
            directory=a.output/arm
            finished=directory/"metrics.csv"
            signature={"arm":arm,**config}
            checkpoint=directory/"checkpoint_last.pt"
            if finished.is_file():
                if not checkpoint.is_file():
                    raise ValueError("M6.7 metrics without final checkpoint")
                saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
                if (saved.get("signature")!=signature or
                    saved.get("completed_epoch")!=20):
                    raise ValueError("Incompatible completed M6.7 arm")
                with finished.open(newline="") as fh:
                    rows=list(csv.DictReader(fh))
                if {row["task"] for row in rows}!=set(TRAIN_TASKS):
                    raise ValueError("Unexpected M6.7 task metrics")
                all_rows.extend(rows)
                print(f"[m67] HIT completed arm={arm}",flush=True)
                continue
            if directory.exists() and any(directory.iterdir()) and not checkpoint.exists():
                raise FileExistsError("Non-resumable arm files: "+str(directory))
            directory.mkdir(parents=True,exist_ok=True)
            torch.set_rng_state(cpu_rng.clone())
            if gpu_rng is not None:
                torch.cuda.set_rng_state(gpu_rng.clone(),device=device)
            model=PredictiveNeighborResidual(
                visual.shape[-1],first.shape[-1],hidden=128,heads=4,
                group_counts=tuple(manifest["group_counts"])).to(device)
            model.load_state_dict(initial,strict=True)
            for p in model.base.parameters():
                p.requires_grad_(False)
            print(f"[m67] START arm={arm} train={len(train_ids)} "
                  f"held={len(held_ids)}",flush=True)
            train_arm(model,arm,visual,geo,neighborhood,graph,oof,
                      y,names,train_ids,stats,cache,directory,signature,a,device)
            rows=[]
            for task in TRAIN_TASKS:
                ix=np.asarray([int(i) for i in held_ids
                               if names[int(i)]==task],dtype=np.int64)
                result=evaluate(
                    model,arm,task,ix,visual,geo,neighborhood,graph,oof,
                    y,names,stats,cache,a.batch_size,device)
                rows.append(result)
                print(f"[m67] {arm:16s} {task:20s} "
                      f"MSE={result['mse_log1p']:.3f} "
                      f"Pearson={result['pearson']:.3f} "
                      f"bias={result['bias']:+.3f}",flush=True)
            write_csv(finished,rows)
            all_rows.extend(rows)
            del model
            if device.type=="cuda":
                torch.cuda.empty_cache()
        write_csv(a.output/"SUMMARY.csv",all_rows)
        lines=[
            "# M6.7 source-city cross-indicator predictive context",
            "",
            "All student arms share frozen DEV visual anchor, initial "
            "trainable weights and source city-fold OOF Population/Food "
            "predictive features. No held city labels were used for fitting.",
            "",
            "| Task | Arm | MSE ↓ | Pearson ↑ | Bias |",
            "|---|---|---:|---:|---:|"]
        for task in TRAIN_TASKS:
            for arm in ARMS:
                row=next(item for item in all_rows if
                         item["task"]==task and item["arm"]==arm)
                lines.append(
                    f"| {task} | {arm} | "
                    f"{float(row['mse_log1p']):.3f} | "
                    f"{float(row['pearson']):.3f} | "
                    f"{float(row['bias']):+.3f} |")
        lines.extend(["",
                      "cross_indicator uses both predicted neighbor A and B; "
                      "same_indicator repeats A; no_predictive has zero "
                      "predicted-neighbor channels. All models retain "
                      "the same physical RemoteCLIP neighborhood.",
                      "OOF nuisance predictors: fixed 4 city-fold "
                      "StandardScaler + Ridge alpha 100, supervised only "
                      "on original source-fit city labels.",
                      "Single-seed source-held-city development evidence; "
                      "not independent target Education/test evaluation."])
        (a.output/"REPORT.md").write_text("\n".join(lines)+"\n")
        print(f"[m67] COMPLETE {a.output}",flush=True)
    finally:
        store.close()


if __name__=="__main__":
    main()
