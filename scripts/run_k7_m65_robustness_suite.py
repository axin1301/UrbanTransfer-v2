"""Unattended frozen M6.5 neighborhood robustness suite, source-only.

Existing DEV checkpoint only, no model fitting and no target/Education labels.
Conditions: real, zero, city_mean, and N independently-seeded derangements
of neighborhood features among physical tiles WITHIN each held source city.
Predictions are checkpointed atomically per condition for safe restart.

Produces concise 3dp metrics, randomized-null summary, per-city paired
loss contrasts, cluster bootstrap CI, and an autonomous Markdown report.
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

from scripts.run_k7_m62_full import CachedTokens,validate_full
from scripts.run_k7_m63_full import PersistentReadOnlyHiddenStore
from scripts.run_k7_m65_full import city_dev_split,fit_source_moments
from scripts.run_k7_m65_dev_intervention import (
    permute_physical_neighborhood,validate_dev_provenance)
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS
from urban_transfer.k7_m65_neighborhood import load_or_build_neighborhood
from urban_transfer.k7_m65_neighborhood_student import NeighborhoodCityIndicatorResidual


def city_mean_neighborhood(frame,nb,held_ids):
    """Retain only city-level context; remove tile-specific neighbor pairing."""
    result=nb.clone()
    cities=frame.city.astype(str).to_numpy()
    tiles=frame.image_name.astype(str).to_numpy()
    ids=np.asarray(held_ids,dtype=np.int64)
    for city in sorted(set(cities[ids])):
        group=ids[cities[ids]==city]
        physical={}
        for ix in group:
            physical.setdefault(tiles[ix],int(ix))
        mean=nb[list(physical.values())].mean(0)
        result[group]=mean
    return result


def stat(pred,truth):
    pred=np.asarray(pred,dtype=np.float64)
    truth=np.asarray(truth,dtype=np.float64)
    err=pred-truth
    return dict(
        n_rows=int(len(pred)),
        mse_log1p=float(np.mean(err**2)),
        mae_log1p=float(np.mean(np.abs(err))),
        pearson=float(np.corrcoef(pred,truth)[0,1])
            if len(pred)>=2 and np.std(pred)>1e-10
            and np.std(truth)>1e-10 else float("nan"),
        mean_bias_log1p=float(np.mean(err)),
        pred_std=float(np.std(pred)),
        truth_std=float(np.std(truth)))


def rows_three_decimals(items):
    return [{k:(f"{v:.3f}" if isinstance(v,(float,np.floating))
                 else v) for k,v in row.items()} for row in items]


def write_csv(path,rows):
    if not rows:
        raise ValueError("No results to write")
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows_three_decimals(rows))
    os.replace(tmp,path)


def read_prediction(path,expected_indices):
    with np.load(path,allow_pickle=False) as data:
        if not np.array_equal(data["ids"],expected_indices):
            raise ValueError("Stale/corrupt prediction indices "+str(path))
        return np.asarray(data["pred"],dtype=np.float64),np.asarray(
            data["route"],dtype=np.float64)


def save_prediction(path,ids,pred,route):
    tmp=path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,ids=np.asarray(ids,dtype=np.int64),
                        pred=np.asarray(pred,dtype=np.float64),
                        route=np.asarray(route,dtype=np.float32))
    os.replace(tmp,path)


@torch.inference_mode()
def predict(model,ids,visual,geo,nb,names,stats,cache,batch_size,device):
    result=[]; route=[]
    for start in range(0,len(ids),batch_size):
        ix=ids[start:start+batch_size]
        h=cache.batch(ix,[names[int(i)] for i in ix],device)
        z,w,*_=model(visual[ix].to(device),geo[ix].to(device),
                     h,nb[ix].to(device),return_aux=True)
        s=np.array([stats[names[int(i)]][1] for i in ix],
                   dtype=np.float64)
        m=np.array([stats[names[int(i)]][0] for i in ix],
                   dtype=np.float64)
        result.extend((m+s*z.cpu().numpy()).tolist())
        route.extend(w[:,0].cpu().numpy().tolist())
    return np.asarray(result,dtype=np.float64),np.asarray(route,dtype=np.float64)


def city_effects(frame,ids,truth,baseline,other,condition,task):
    cities=frame.city.astype(str).to_numpy()[ids]
    rows=[]
    for city in sorted(set(cities)):
        ix=np.flatnonzero(cities==city)
        y=truth[ix]; a=baseline[ix]; b=other[ix]
        real_mse=float(np.mean((a-y)**2))
        test_mse=float(np.mean((b-y)**2))
        rows.append(dict(task=task,condition=condition,city=city,
                         n_rows=len(ix),real_mse=real_mse,
                         perturbed_mse=test_mse,
                         perturbed_minus_real=test_mse-real_mse,
                         real_pearson=stat(a,y)["pearson"],
                         perturbed_pearson=stat(b,y)["pearson"]))
    return rows


def bootstrap_city_effect(city_rows,reps=2000,seed=8801):
    """Cluster bootstrap: resample city-level loss numerators and counts."""
    n=np.asarray([r["n_rows"] for r in city_rows],dtype=float)
    delta=np.asarray([r["perturbed_minus_real"] for r in city_rows],dtype=float)
    if not len(city_rows):
        raise ValueError("No cities to bootstrap")
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(city_rows),size=(reps,len(city_rows)))
    sampled_n=n[draws]
    effect=np.sum(sampled_n*delta[draws],axis=1)/sampled_n.sum(axis=1)
    return float(np.quantile(effect,.025)),float(np.quantile(effect,.975))


def summarize(frame,ids,predictions,truth,n_permutations):
    city_rows=[]; metrics=[]; summary=[]
    real=predictions["real"][0]
    for task in TRAIN_TASKS:
        task_ix=np.flatnonzero(
            np.asarray(frame.indicator_column.astype(str).to_numpy()[ids])==task)
        task_ids=ids[task_ix]
        truth_task=truth[task_ix]
        baseline=real[task_ix]
        for condition,(pred,route) in predictions.items():
            s=stat(pred[task_ix],truth_task)
            metrics.append(dict(task=task,condition=condition,**s,
                                population_weight=float(np.mean(route[task_ix])),
                                prediction_mad_vs_real=float(np.mean(
                                    np.abs(pred[task_ix]-baseline)))))
            if condition!="real":
                city_rows.extend(city_effects(
                    frame,task_ids,truth_task,baseline,pred[task_ix],
                    condition,task))
        base=stat(baseline,truth_task)
        subset=[r for r in metrics if r["task"]==task]
        for mode in ("zero","city_mean"):
            row=next(x for x in subset if x["condition"]==mode)
            cities=[x for x in city_rows if x["task"]==task and
                    x["condition"]==mode]
            lo,hi=bootstrap_city_effect(cities)
            summary.append({
                "task":task,"comparison":mode,"n_trials":1,
                "real_mse":base["mse_log1p"],
                "perturbed_mse_mean":row["mse_log1p"],
                "perturbed_minus_real_mse_mean":
                    row["mse_log1p"]-base["mse_log1p"],
                "city_bootstrap_ci_low":lo,"city_bootstrap_ci_high":hi,
                "real_pearson":base["pearson"],
                "perturbed_pearson_mean":row["pearson"],
                "real_minus_perturbed_pearson_mean":
                    base["pearson"]-row["pearson"],
                "fraction_held_cities_worse":float(np.mean(
                    [x["perturbed_minus_real"]>0 for x in cities])),
                "permutation_p_one_sided":float("nan")})
        perm_metrics=[x for x in subset if x["condition"].startswith("perm_")]
        effects=np.array([x["mse_log1p"]-base["mse_log1p"]
                          for x in perm_metrics])
        pearson_deltas=np.array([base["pearson"]-x["pearson"]
                                 for x in perm_metrics])
        # Predeclared one-sided randomization p with +1 correction:
        # how often permuted had <= real MSE, i.e. real no better.
        p=(1+np.sum(effects<=0))/(len(effects)+1)
        perm_cities=[x for x in city_rows
                     if x["task"]==task and x["condition"].startswith("perm_")]
        pooled={}
        for row in perm_cities:
            pooled.setdefault(row["city"],[]).append(row)
        city_avgs=[]
        for city,values in pooled.items():
            city_avgs.append({
                "city":city,"n_rows":values[0]["n_rows"],
                "perturbed_minus_real":float(np.mean(
                    [x["perturbed_minus_real"] for x in values]))})
        lo,hi=bootstrap_city_effect(city_avgs)
        summary.append({
            "task":task,"comparison":"in_city_tile_derangement",
            "n_trials":n_permutations,
            "real_mse":base["mse_log1p"],
            "perturbed_mse_mean":float(np.mean(
                [r["mse_log1p"] for r in perm_metrics])),
            "perturbed_minus_real_mse_mean":float(effects.mean()),
            "city_bootstrap_ci_low":lo,"city_bootstrap_ci_high":hi,
            "real_pearson":base["pearson"],
            "perturbed_pearson_mean":float(np.mean(
                [r["pearson"] for r in perm_metrics])),
            "real_minus_perturbed_pearson_mean":
                float(pearson_deltas.mean()),
            "fraction_held_cities_worse":float(np.mean([
                r["perturbed_minus_real"]>0 for r in city_avgs])),
            "permutation_p_one_sided":float(p)})
    return metrics,city_rows,summary


def report_markdown(summary,number):
    lines=[
        "# M6.5 DEV frozen-neighborhood robustness report",
        "",
        "Fixed DEV checkpoint, Population/Food held-source cities only. "
        "No training and no Education/official-test outcomes.",
        "",
        "Positive ΔMSE means the intervention is **worse** than real "
        "neighborhoods. Positive ΔPearson means real is better.",
        "",
        "| Task | Intervention | Trials | Real MSE | Intervention MSE | "
        "ΔMSE | City-bootstrap 95% CI of ΔMSE | "
        "Real Pearson | Intervention Pearson | ΔPearson | Cities worse |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            "| {task} | {comparison} | {n_trials} | {real_mse:.3f} | "
            "{perturbed_mse_mean:.3f} | "
            "{perturbed_minus_real_mse_mean:+.3f} | "
            "[{city_bootstrap_ci_low:+.3f}, {city_bootstrap_ci_high:+.3f}] | "
            "{real_pearson:.3f} | {perturbed_pearson_mean:.3f} | "
            "{real_minus_perturbed_pearson_mean:+.3f} | "
            "{fraction_held_cities_worse:.3f} |".format(**r))
    lines.extend([
        "",
        f"Repeated within-city tile-derangement: {number} seeds.",
        "The one-sided randomization rank p = (1 + #permutations whose "
        "MSE <= real MSE)/(N+1); minimal possible p is 1/(N+1).",
        "This is a **conditional perturbation diagnostic**, not an "
        "independent sampling-based p-value or a causal claim.",
        "City-bootstrap intervals resample the *held source cities* "
        "after averaging perturbation effects per city, not individual "
        "rows. Few cities yield unstable intervals.",
        "City mean removes tile-specific pairing while keeping one "
        "same-city aggregate neighborhood vector (an intervention, "
        "not a separately trained baseline).",
        "Zero is an out-of-training-distribution stress test.",
        "Existing retrained zero-neighborhood ablation gave mixed "
        "results; do not claim general benefit from interventions alone.",
        "",
        "## Interpretation guardrails",
        "- These results measure the frozen network's sensitivity to "
        "neighbor information and correct spatial matching.",
        "- They do not establish that true neighbors outperform a "
        "separately retrained model on source-held cities.",
        "- The UCUI Education development labels and official test "
        "labels were neither loaded nor used for selection.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-dir",type=Path,
                        default=Path("runs/k7/m62_full_seed42"))
    parser.add_argument("--dev-dir",type=Path,
                        default=Path("runs/k7/m65_neighborhood_dev_seed42"))
    parser.add_argument("--neighbor-cache",type=Path,
                        default=Path("runs/k7/m65_neighbors_k8.pt"))
    parser.add_argument("--output",type=Path,
                        default=Path("runs/k7/m65_dev_robustness_seed42"))
    parser.add_argument("--permutations",type=int,default=20)
    parser.add_argument("--seed-start",type=int,default=10001)
    parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    a=parser.parse_args()
    if not 2<=a.permutations<=200 or a.batch_size<=0:
        parser.error("2-200 permutations and positive batch-size required")
    if not torch.cuda.is_available() and a.device=="cuda":
        raise RuntimeError("GPU unavailable")
    if a.output.resolve()==a.dev_dir.resolve():
        raise ValueError("Never overwrite the trained M6.5 DEV directory")

    manifest_path=(a.full_dir/"full_manifest.json").resolve()
    manifest=json.loads(manifest_path.read_text())
    frame,x,geo,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    splits=validate_full(frame,manifest)
    saved,fit_ids,held_ids,cities,stats=validate_dev_provenance(
        a.dev_dir,manifest_path,frame,splits,y,names)
    nb=load_or_build_neighborhood(frame,x,manifest_path,
                                  a.neighbor_cache,k=8)
    if len(held_ids)<2:
        raise ValueError("Source-held city split is empty")
    config={
        "version":1,"manifest":str(manifest_path),
        "dev_checkpoint":str((a.dev_dir/"m65.pt").resolve()),
        "dev_checkpoint_sha256":hashlib.sha256(
            (a.dev_dir/"m65.pt").read_bytes()).hexdigest(),
        "source_train_hash":hashlib.sha256(
            np.asarray(fit_ids,dtype=np.int64).tobytes()).hexdigest(),
        "held_index_hash":hashlib.sha256(
            np.asarray(held_ids,dtype=np.int64).tobytes()).hexdigest(),
        "permutations":a.permutations,"seed_start":a.seed_start,
        "held_cities":cities,"conditions":["real","zero","city_mean",
            "perm_000..."],"no_retraining":True,
        "education_outcomes_read":False,
        "official_test_outcomes_read":False
    }
    a.output.mkdir(parents=True,exist_ok=True)
    config_path=a.output/"config.json"
    if config_path.exists():
        if json.loads(config_path.read_text())!=config:
            raise ValueError("Robustness output config differs; use another --output")
    else:
        if any(a.output.iterdir()):
            raise FileExistsError("Unrecognized non-empty robustness output directory")
        config_path.write_text(json.dumps(config,indent=2))
    pred_dir=a.output/"predictions"
    pred_dir.mkdir(exist_ok=True)

    ds=ImageUrbanDataset(
        manifest["source_csv"],manifest["image_root"],partition="eval")
    if len(ds.df)!=len(frame) or names!=ds.df.indicator_column.astype(str).tolist():
        raise ValueError("Token and tensor rows misaligned")
    store=PersistentReadOnlyHiddenStore(a.full_dir/"hidden.sqlite3")
    try:
        cache=CachedTokens(frame,ds,manifest,store,maxsize=512)
        first=cache[int(fit_ids[0]),names[int(fit_ids[0])]]
        model=NeighborhoodCityIndicatorResidual(
            x.shape[-1],first.shape[-1],hidden=128,heads=4,
            group_counts=tuple(manifest["group_counts"]),
            max_residual=.5)
        model.load_state_dict(saved["model"],strict=True)
        model=model.to(a.device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        conditions=["real","zero","city_mean"]+[
            f"perm_{i:03d}" for i in range(a.permutations)]
        predictions={}
        for ci,condition in enumerate(conditions):
            target=pred_dir/(condition+".npz")
            if target.is_file():
                prediction,route=read_prediction(target,held_ids)
                print(f"[m65-robust] HIT {condition} "
                      f"({ci+1}/{len(conditions)})",flush=True)
            else:
                started=time.monotonic()
                if condition=="real":
                    features=nb
                elif condition=="zero":
                    features=torch.zeros_like(nb)
                elif condition=="city_mean":
                    features=city_mean_neighborhood(frame,nb,held_ids)
                else:
                    seed=a.seed_start+int(condition.split("_")[1])
                    features=permute_physical_neighborhood(
                        frame,nb,held_ids,seed=seed)
                prediction,route=predict(
                    model,held_ids,x,geo,features,names,stats,cache,
                    a.batch_size,a.device)
                save_prediction(target,held_ids,prediction,route)
                print(f"[m65-robust] SAVED {condition} "
                      f"({ci+1}/{len(conditions)}) "
                      f"elapsed_s={time.monotonic()-started:.1f}",flush=True)
            predictions[condition]=(prediction,route)
            if condition=="real":
                with (a.dev_dir/"metrics.csv").open(newline="") as fh:
                    original={row["partition"]:row for row in csv.DictReader(fh)}
                if set(original)!={"dev_held_city_"+t for t in TRAIN_TASKS}:
                    raise ValueError("Original DEV metrics unexpectedly include targets")
                for task in TRAIN_TASKS:
                    task_mask=np.asarray(names,dtype=object)[held_ids]==task
                    evaluated=stat(prediction[task_mask],
                                   y[torch.as_tensor(held_ids[task_mask])].numpy())
                    benchmark=original["dev_held_city_"+task]
                    for metric in ("mse_log1p","pearson","mean_bias_log1p"):
                        if not np.isclose(evaluated[metric],float(benchmark[metric]),
                                          atol=1e-4,rtol=1e-4):
                            raise ValueError("Real dev reproduction failed: "+
                                             task+"/"+metric)
        truth=y[torch.as_tensor(held_ids)].numpy().astype(np.float64)
        metrics,city_rows,summary=summarize(
            frame,held_ids,predictions,truth,a.permutations)
        write_csv(a.output/"metrics.csv",metrics)
        write_csv(a.output/"per_city.csv",city_rows)
        write_csv(a.output/"summary.csv",summary)
        (a.output/"REPORT.md").write_text(
            report_markdown(summary,a.permutations))
        print("[m65-robust] ===== SUMMARY (3dp) =====",flush=True)
        for row in summary:
            print(f"[m65-robust] {row['task']:20s} "
                  f"{row['comparison']:26s} "
                  f"dMSE={row['perturbed_minus_real_mse_mean']:+.3f} "
                  f"CI95=[{row['city_bootstrap_ci_low']:+.3f},"
                  f"{row['city_bootstrap_ci_high']:+.3f}] "
                  f"dPearson={row['real_minus_perturbed_pearson_mean']:+.3f} "
                  f"cities_worse={row['fraction_held_cities_worse']:.3f}",
                  flush=True)
        print(f"[m65-robust] COMPLETE {a.output}",flush=True)
    finally:
        store.close()


if __name__=="__main__":
    main()
