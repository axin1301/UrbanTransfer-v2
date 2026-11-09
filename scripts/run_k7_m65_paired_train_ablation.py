"""Strictly paired M6.5 DEV training ablation: Real / Zero / City-mean.

All three arms use SAME freshly constructed trainable initial weights,
identical saved source-only DEV visual anchor, fixed city-held source
partition, 20 epochs, seed42 batch order and training losses.
Each arm trains independently. Existing M6.5 outputs are never altered.
Only source held-city Pop/Food labels are evaluated. Unlabeled image
features for city means may include tiles from held-city partitions.
No target Education outcomes, no official test file access.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts.run_k7_m62_full import CachedTokens,validate_full
from scripts.run_k7_m63_full import PersistentReadOnlyHiddenStore
from scripts.run_k7_m65_full import (
    train_neighbor_student,evaluate_source_or_target)
from scripts.run_k7_m65_dev_intervention import validate_dev_provenance
from scripts.run_k7_m65_robustness_suite import (
    city_mean_neighborhood,write_csv)
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS
from urban_transfer.k7_m65_neighborhood import load_or_build_neighborhood
from urban_transfer.k7_m65_neighborhood_student import (
    NeighborhoodCityIndicatorResidual)


ARMS=("real","zero","city_mean")


def build_arms(frame,features,ids):
    return {
        "real":features,
        "zero":torch.zeros_like(features),
        "city_mean":city_mean_neighborhood(frame,features,ids),
    }


def summarize(arm_metrics):
    output=[]
    for task in TRAIN_TASKS:
        part="dev_held_city_"+task
        indexed={arm:next(r for r in arm_metrics[arm]
                          if r["partition"]==part) for arm in ARMS}
        real=indexed["real"]
        for arm in ARMS:
            row=indexed[arm]
            output.append({
                "task":task,"arm":arm,"n_rows":int(row["n_rows"]),
                "mse_log1p":float(row["mse_log1p"]),
                "pearson":float(row["pearson"]),
                "mae_log1p":float(row["mae_log1p"]),
                "bias":float(row["mean_bias_log1p"]),
                "pred_std":float(row["pred_std"]),
                "population_weight":float(row["population_weight"]),
                "real_minus_arm_mse":float(real["mse_log1p"]-
                                            row["mse_log1p"]),
                "real_minus_arm_pearson":float(real["pearson"]-
                                                row["pearson"]),
            })
    return output


def ensure_run_config(directory,config):
    directory.mkdir(parents=True,exist_ok=True)
    cfg=directory/"config.json"
    if cfg.exists():
        if json.loads(cfg.read_text())!=config:
            raise ValueError("Existing paired ablation config mismatch")
    else:
        if any(directory.iterdir()):
            raise FileExistsError(
                "Output exists with unrelated files: "+str(directory))
        tmp=directory/"config.json.tmp"
        tmp.write_text(json.dumps(config,indent=2))
        os.replace(tmp,cfg)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-dir",type=Path,
                   default=Path("runs/k7/m62_full_seed42"))
    p.add_argument("--dev-dir",type=Path,
                   default=Path("runs/k7/m65_neighborhood_dev_seed42"))
    p.add_argument("--neighbor-cache",type=Path,
                   default=Path("runs/k7/m65_neighbors_k8.pt"))
    p.add_argument("--output",type=Path,
                   default=Path("runs/k7/m65_paired_ablation_dev_seed42"))
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    a=p.parse_args()
    if a.epochs!=20 or a.batch_size!=256:
        p.error("Fixed 20 epochs / batch256 for paired M6.5 protocol")
    if a.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    if a.output.resolve()==a.dev_dir.resolve():
        raise ValueError("Paired output must not overwrite original DEV")
    if (a.output/"SUMMARY.csv").exists():
        print(f"[m65-paired] ALREADY COMPLETE {a.output}",flush=True)
        return

    manifest_path=(a.full_dir/"full_manifest.json").resolve()
    manifest=json.loads(manifest_path.read_text())
    frame,x,ctx,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    splits=validate_full(frame,manifest)
    dev_saved,fit_ids,held_ids,cities,stats=validate_dev_provenance(
        a.dev_dir,manifest_path,frame,splits,y,names)
    anchor_path=a.dev_dir/"dev_visual_anchor.pt"
    anchor=torch.load(anchor_path,map_location="cpu",weights_only=True)
    anchor_sha=hashlib.sha256(anchor_path.read_bytes()).hexdigest()
    source_sha=hashlib.sha256(np.asarray(
        fit_ids,dtype=np.int64).tobytes()).hexdigest()
    held_sha=hashlib.sha256(np.asarray(
        held_ids,dtype=np.int64).tobytes()).hexdigest()
    actual=load_or_build_neighborhood(
        frame,x,manifest_path,a.neighbor_cache,k=8)
    # City means are calculated from unlabeled physical tiles across
    # ALL source CSV rows (same transductive scope as real KNN), not
    # from task labels. Only per-city means are used; duplicates deduped.
    neighborhoods=build_arms(
        frame,actual,np.arange(len(frame),dtype=np.int64))
    run_config={
        "protocol":"paired_m65_dev_real_zero_city_mean_v1",
        "manifest":str(manifest_path),
        "source_train_sha256":source_sha,
        "held_source_sha256":held_sha,
        "held_source_cities":cities,
        "dev_anchor_sha256":anchor_sha,
        "dev_checkpoint_sha256":hashlib.sha256(
            (a.dev_dir/"m65.pt").read_bytes()).hexdigest(),
        "neighbor_cache":str(a.neighbor_cache.resolve()),
        "arms":list(ARMS),"epochs":20,"batch_size":256,
        "initial_seed":42,"shuffle_seed":42,
        "education_labels_used":False,
        "official_test_labels_used":False,
        "city_mean_scope":"all source_csv unlabeled physical tiles",
    }
    ensure_run_config(a.output,run_config)

    ds=ImageUrbanDataset(
        manifest["source_csv"],manifest["image_root"],partition="eval")
    if len(ds.df)!=len(frame) or names!=ds.df.indicator_column.astype(str).tolist():
        raise ValueError("Token/tensor source row mismatch")
    store=PersistentReadOnlyHiddenStore(a.full_dir/"hidden.sqlite3")
    try:
        cache=CachedTokens(frame,ds,manifest,store,maxsize=512)
        first=cache[int(fit_ids[0]),names[int(fit_ids[0])]]
        device=torch.device(a.device)

        # Explicitly initialize ONCE and restore the identical trainable
        # weights and RNG states for each arm's training. The frozen
        # anchor is exactly equal across arms by construction.
        torch.manual_seed(42)
        if device.type=="cuda":
            torch.cuda.manual_seed_all(42)
        common=NeighborhoodCityIndicatorResidual(
            x.shape[-1],first.shape[-1],hidden=128,heads=4,
            group_counts=tuple(manifest["group_counts"]),
            max_residual=.5)
        common.load_visual_baseline(anchor["model"])
        for parameter in common.base.parameters():
            parameter.requires_grad_(False)
        initial={k:v.detach().cpu().clone()
                 for k,v in common.state_dict().items()}
        cpu_state=torch.get_rng_state().clone()
        gpu_state=(torch.cuda.get_rng_state(device=device).clone()
                   if device.type=="cuda" else None)
        del common
        arm_metrics={}
        for arm in ARMS:
            out=a.output/arm
            checkpoint=out/"checkpoint_last.pt"
            finished=out/"metrics.csv"
            if finished.is_file():
                # After a restart verify the saved output matches protocol
                # and has a final checkpoint before trusting metrics.
                cfg_file=out/"arm_config.json"
                if (not checkpoint.is_file() or not cfg_file.is_file() or
                    json.loads(cfg_file.read_text())!={"arm":arm,
                        **run_config}):
                    raise ValueError("Existing completed arm metadata mismatch")
                import csv
                with finished.open(newline="") as handle:
                    arm_metrics[arm]=list(csv.DictReader(handle))
                print(f"[m65-paired] HIT completed arm={arm}",flush=True)
                continue
            if out.exists() and any(out.iterdir()) and not checkpoint.is_file():
                raise FileExistsError(
                    "Arm has files without resumable checkpoint: "+str(out))
            out.mkdir(parents=True,exist_ok=True)

            torch.set_rng_state(cpu_state.clone())
            if gpu_state is not None:
                torch.cuda.set_rng_state(gpu_state.clone(),device=device)
            model=NeighborhoodCityIndicatorResidual(
                x.shape[-1],first.shape[-1],hidden=128,heads=4,
                group_counts=tuple(manifest["group_counts"]),
                max_residual=.5).to(device)
            model.load_state_dict(initial,strict=True)
            for parameter in model.base.parameters():
                parameter.requires_grad_(False)
            signature={"arm":arm,**run_config}
            print(f"[m65-paired] START arm={arm} "
                  f"train={len(fit_ids)} held={len(held_ids)}",
                  flush=True)
            train_neighbor_student(
                model,x,ctx,neighborhoods[arm],y,names,fit_ids,
                stats,cache,a,device,checkpoint,signature)
            model.eval()
            metrics=[]
            for task in TRAIN_TASKS:
                ix=np.asarray([int(i) for i in held_ids
                               if names[int(i)]==task],dtype=np.int64)
                result=evaluate_source_or_target(
                    model,"dev_held_city_"+task,ix,x,ctx,
                    neighborhoods[arm],y,names,stats,(0.,1.),
                    cache,a.batch_size,device)
                result["model"]="m65_paired_"+arm
                metrics.append(result)
                print(f"[m65-paired] {arm:10s} {task:20s} "
                      f"MSE={result['mse_log1p']:.3f} "
                      f"Pearson={result['pearson']:.3f} "
                      f"bias={result['mean_bias_log1p']:+.3f}",
                      flush=True)
            write_csv(finished,metrics)
            (out/"arm_config.json").write_text(
                json.dumps(signature,indent=2))
            arm_metrics[arm]=metrics
            del model
            if device.type=="cuda":
                torch.cuda.empty_cache()

        results=summarize(arm_metrics)
        write_csv(a.output/"SUMMARY.csv",results)
        lines=[
            "# M6.5 DEV strictly paired source-city training ablation",
            "",
            "Same frozen DEV anchor, trainable initial state, model "
            "architecture, train cities, batch permutations and 20-epoch "
            "losses; only the neighborhood input differs.",
            "",
            "| Task | Arm | MSE ↓ | Pearson ↑ | Bias | "
            "Real minus arm MSE | Real minus arm Pearson |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in results:
            lines.append(
                f"| {row['task']} | {row['arm']} | "
                f"{row['mse_log1p']:.3f} | {row['pearson']:.3f} | "
                f"{row['bias']:+.3f} | "
                f"{row['real_minus_arm_mse']:+.3f} | "
                f"{row['real_minus_arm_pearson']:+.3f} |")
        lines.extend([
            "",
            "MSE: negative real-minus-arm means real neighbors are "
            "better. Pearson: positive means real neighbors are better.",
            "This is a single-seed paired initialization/training-budget "
            "comparison, not proof of broad reproducibility. Education "
            "labels and official test outcomes remain unopened.",
            "City-mean features are transductive from unlabeled source-CSV "
            "physical tiles, as in original real KNN pipeline.",
        ])
        (a.output/"REPORT.md").write_text("\n".join(lines)+"\n")
        print(f"[m65-paired] COMPLETE {a.output}",flush=True)
    finally:
        store.close()


if __name__=="__main__":
    main()
