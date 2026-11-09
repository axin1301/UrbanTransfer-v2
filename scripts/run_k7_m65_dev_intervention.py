"""Frozen M6.5 DEV neighborhood sensitivity on held-source cities.

Three conditions on identical pretrained dev student:
  real: original K=8 physical-neighbor features
  zero: all-zero input (out-of-distribution stress test)
  city_permuted: different physical tile's neighbor vector, within city,
                 stable across duplicated indicator rows (spatial pairing test)
No retraining, no Education labels, no original UCSI or official test data.
No SCUI/UCUI target labels used for design, scoring or selection.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts.run_k7_m62_full import CachedTokens,validate_full
from scripts.run_k7_m63_full import PersistentReadOnlyHiddenStore
from scripts.run_k7_m65_full import (
    city_dev_split,fit_source_moments,evaluate_source_or_target)
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS
from urban_transfer.k7_m65_neighborhood import load_or_build_neighborhood
from urban_transfer.k7_m65_neighborhood_student import NeighborhoodCityIndicatorResidual


def permute_physical_neighborhood(frame,neighbors,held_ids,seed=4265):
    """In-city physical-tile derangement, with no indicator/label dependency.

    Only tile identity and city are consulted; the same tile repeated
    for source tasks gets exactly the same donor's neighbor vector.
    All other row features remain unchanged, including model visuals.
    """
    ids=np.asarray(held_ids,dtype=np.int64)
    city=frame.city.astype(str).to_numpy()
    tile=frame.image_name.astype(str).to_numpy()
    result=neighbors.clone()
    rng=np.random.default_rng(seed)
    for c in sorted(set(city[ids])):
        rows=ids[city[ids]==c]
        key_to_rows={}
        for i in rows:
            key_to_rows.setdefault(tile[int(i)],[]).append(int(i))
        names=sorted(key_to_rows)
        if len(names)<2:
            raise ValueError("Need >=2 unique physical held tiles in city "+c)
        # Random cyclic permutation guarantees no same-tile donors.
        shuffled=rng.permutation(len(names))
        for pos in range(len(names)):
            recipient=names[int(shuffled[pos])]
            donor=names[int(shuffled[(pos+1)%len(names)])]
            donor_idx=key_to_rows[donor][0]
            for i in key_to_rows[recipient]:
                result[i]=neighbors[donor_idx]
    return result


def validate_dev_provenance(dev_dir,manifest_path,frame,splits,y,names):
    fit_ids,held_ids,cities=city_dev_split(frame,splits["train"])
    source_hash=hashlib.sha256(
        np.asarray(fit_ids,dtype=np.int64).tobytes()).hexdigest()
    stats=fit_source_moments(y,names,fit_ids)
    cfg=json.loads((dev_dir/"config.json").read_text())
    if (cfg.get("stage")!="dev" or
        Path(cfg["manifest"]).resolve()!=manifest_path.resolve() or
        cfg["source_train_sha256"]!=source_hash or
        cfg["held_source_cities"]!=cities or
        int(cfg["k"])!=8):
        raise ValueError("Dev city split or source-only manifest mismatch")
    saved=torch.load(dev_dir/"m65.pt",
                     map_location="cpu",weights_only=True)
    if saved["config"]!={key:cfg[key] for key in saved["config"]}:
        raise ValueError("Dev student configuration mismatch")
    if saved["source_stats"]!=stats:
        raise ValueError("Dev training moments mismatch")
    anchor=torch.load(dev_dir/"dev_visual_anchor.pt",
                      map_location="cpu",weights_only=True)
    if (anchor["signature"]["train_hash"]!=source_hash or
        Path(anchor["signature"]["manifest"]).resolve()!=
        manifest_path.resolve()):
        raise ValueError("Dev anchor source city split mismatch")
    for name,value in anchor["model"].items():
        if not torch.equal(saved["model"]["base."+name],value):
            raise ValueError("M6.5 student does not contain original dev anchor")
    return saved,fit_ids,held_ids,cities,stats


def compare_rows(all_rows):
    grouped={}
    for row in all_rows:
        grouped.setdefault(row["partition"],{})[row["condition"]]=row
    records=[]
    for partition,conditions in grouped.items():
        base=conditions["real"]
        for condition in ("zero","city_permuted"):
            candidate=conditions[condition]
            for metric in ("mse_log1p","pearson","mae_log1p",
                           "mean_bias_log1p","pred_std","population_weight"):
                records.append({
                    "partition":partition,"condition":condition,
                    "metric":metric,
                    "real":float(base[metric]),
                    "perturbed":float(candidate[metric]),
                    "real_minus_perturbed":
                        float(base[metric])-float(candidate[metric])
                })
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-dir",type=Path,
                   default=Path("runs/k7/m62_full_seed42"))
    p.add_argument("--dev-dir",type=Path,
                   default=Path("runs/k7/m65_neighborhood_dev_seed42"))
    p.add_argument("--neighbor-cache",type=Path,
                   default=Path("runs/k7/m65_neighbors_k8.pt"))
    p.add_argument("--output",type=Path,
                   default=Path("runs/k7/m65_dev_neighborhood_intervention"))
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=p.parse_args()
    if args.batch_size<1:
        p.error("Inference batch size must be positive")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("No GPU available")
    if args.output.exists():
        raise FileExistsError("Choose a fresh output directory for intervention")
    manifest_path=(args.full_dir/"full_manifest.json").resolve()
    manifest=json.loads(manifest_path.read_text())
    print("[k7-m65-intervene] reading frozen source tensors...",flush=True)
    frame,x,ctx,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    splits=validate_full(frame,manifest)
    saved,fit_ids,held_ids,cities,stats=validate_dev_provenance(
        args.dev_dir,manifest_path,frame,splits,y,names)
    actual=load_or_build_neighborhood(
        frame,x,manifest_path,args.neighbor_cache,k=8)
    perturbed=permute_physical_neighborhood(frame,actual,held_ids)
    conditions={
        "real":actual,
        "zero":torch.zeros_like(actual),
        "city_permuted":perturbed,
    }
    ds=ImageUrbanDataset(
        manifest["source_csv"],manifest["image_root"],partition="eval")
    if len(ds.df)!=len(frame) or names!=ds.df.indicator_column.astype(str).tolist():
        raise ValueError("Tensor rows and source image tasks mismatch")
    store=PersistentReadOnlyHiddenStore(args.full_dir/"hidden.sqlite3")
    try:
        cache=CachedTokens(frame,ds,manifest,store,maxsize=512)
        sample=cache[int(fit_ids[0]),names[int(fit_ids[0])]]
        model=NeighborhoodCityIndicatorResidual(
            x.shape[-1],sample.shape[-1],hidden=128,heads=4,
            group_counts=tuple(manifest["group_counts"]),
            max_residual=.5)
        model.load_state_dict(saved["model"],strict=True)
        model.to(args.device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        original={}
        with (args.dev_dir/"metrics.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                original[row["partition"]]=row
        expected={"dev_held_city_"+t for t in TRAIN_TASKS}
        if set(original)!=expected:
            raise ValueError("Original dev results must be source-task only")
        metrics=[]
        for condition,nb in conditions.items():
            print(f"[k7-m65-intervene] condition={condition} START",flush=True)
            for task in TRAIN_TASKS:
                ids=np.asarray([int(i) for i in held_ids
                                if names[int(i)]==task],dtype=np.int64)
                part="dev_held_city_"+task
                started=time.monotonic()
                result=evaluate_source_or_target(
                    model,part,ids,x,ctx,nb,y,names,stats,
                    (0.,1.),cache,args.batch_size,args.device)
                result["condition"]=condition
                if condition=="real":
                    old=original[part]
                    if int(old["n_rows"])!=len(ids):
                        raise ValueError("Held source row count changed")
                    for key in ("mse_log1p","pearson",
                                "mean_bias_log1p"):
                        if not np.isclose(float(old[key]),float(result[key]),
                                          atol=1e-4,rtol=1e-4):
                            raise ValueError(
                                "Real neighborhood failed original dev "
                                "reproduction: "+part+" / "+key)
                metrics.append(result)
                print(f"[k7-m65-intervene] {condition:14s} {task:22s} "
                      f"n={len(ids)} MSE={result['mse_log1p']:.4f} "
                      f"Pearson={result['pearson']:.4f} "
                      f"bias={result['mean_bias_log1p']:+.4f} "
                      f"pred_std={result['pred_std']:.4f} "
                      f"elapsed_s={time.monotonic()-started:.1f}",flush=True)
        comparison=compare_rows(metrics)
        args.output.mkdir(parents=True,exist_ok=False)
        with (args.output/"metrics.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        with (args.output/"comparison.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
        (args.output/"config.json").write_text(json.dumps({
            "experiment":"m65_frozen_dev_neighborhood_intervention_v1",
            "manifest":str(manifest_path),
            "dev_model":str((args.dev_dir/"m65.pt").resolve()),
            "held_source_cities":cities,
            "held_source_rows":len(held_ids),
            "permutation_seed":4265,
            "conditions":list(conditions),
            "fit_or_calibration_performed":False,
            "target_education_labels_accessed":False,
            "official_test_files_accessed":False,
            "interpretation":"Frozen-input intervention; zero may be "
            "out of distribution. No causal attribution from this alone.",
        },indent=2))
        print(f"[k7-m65-intervene] COMPLETE {args.output}",flush=True)
    finally:
        store.close()


if __name__=="__main__":
    main()
