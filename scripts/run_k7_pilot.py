"""Unattended Z1 semantics/relation experiments with resumable source-only folds.

Run one condition/seed per command. No true target CSV is ever loaded.
Stores each finished fold before continuing; fixed budgets, no UCUI tuning.
"""
import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from scripts.run_z1_pilot import (
    calibrated_moments, fold_indices, scores, tensorize)
from urban_transfer.ktrans_text import qwen_task_vectors
from urban_transfer.m3_features import CachedFeatureDataset, FeatureStore
from urban_transfer.protocol import SOURCE_TASKS
from urban_transfer.z1_model import Z1Regressor
from urban_transfer.k7_hyper import K7HyperRegressor
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.z1_relation import RelationRegressor, fold_relations, relation_embeddings

VARIANTS = ("k7_e1", "k7_e2", "k7_e2_shuffle")
VERSION = "k7-hyper-source-only-v1"


def get_semantics(a):
    profile = "short"  # Freeze scale and predictor semantics across E1/E2
    return qwen_task_vectors(root=a.text_cache_dir, model_id=a.text_model,
                             profile=profile, only_cached=True)


def load_data(a):
    shared = os.environ.get("K7_SHARED_TENSOR_CACHE")
    if shared:
        print(f"[k7] reading shared tensor cache {shared}", flush=True)
        return read_tensor_cache(shared, a.source_csv, a.image_root,
                                 a.feature_dir, a.remoteclip_model, a.image_column)
    # Standalone invocations remain supported, with visible loading progress.
    from urban_transfer.k7_feature_cache import build_tensor_cache
    fallback = a.output.parent / "_k7_source_tensors.pt"
    build_tensor_cache(a.source_csv, a.image_root, a.feature_dir,
                       a.remoteclip_model, fallback, a.image_column)
    return read_tensor_cache(fallback, a.source_csv, a.image_root,
                             a.feature_dir, a.remoteclip_model, a.image_column)


def diagnostic(predicted, truth):
    """Label-accessing ORACLE diagnostic, never a model-selection metric."""
    p = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    bias = float(p.mean() - y.mean())
    pstd, ystd = float(p.std()), float(y.std())
    ratio = pstd / max(ystd, 1e-12)
    if pstd <= 1e-10 or ystd <= 1e-10:
        oracle = float("nan")
    else:
        # Fit affine calibration on the EVALUATION labels. Diagnostic only.
        slope = float(np.cov(p, y, ddof=0)[0, 1] / (pstd ** 2))
        intercept = float(y.mean() - slope * p.mean())
        oracle = float(np.mean((slope * p + intercept - y) ** 2))
    return {"prediction_bias_oracle": bias, "pred_std_to_true_std_oracle": ratio,
            "affine_fitted_mse_ORACLE_NOT_ZERO_SHOT": oracle}


def fold_run(a, task, frame, x, context, y, names, semantics, relation):
    cities = sorted(frame.city.astype(str).unique())
    gen = np.random.default_rng(a.seed)
    shuffled = gen.permutation(cities)
    num = max(1, min(len(cities)-2, round(len(cities)*a.city_fraction)))
    held_cities = sorted(shuffled[:num].tolist())
    splits = fold_indices(frame, task, held_cities)
    stats, unseen = calibrated_moments(
        frame.iloc[splits["train"]], semantics, task, a.scale_temperature)
    train_idx = splits["train"]
    train_mean = torch.tensor([stats[names[i]][0] for i in train_idx],
                              dtype=torch.float32)
    train_std = torch.tensor([stats[names[i]][1] for i in train_idx],
                             dtype=torch.float32)
    target = (y[train_idx] - train_mean) / train_std
    sv = torch.stack([semantics[n].float() for n in names])
    use_relation = a.variant != "k7_e1"
    rv = None
    if use_relation:
        rf = fold_relations(relation, task, SOURCE_TASKS, semantics,
                            shuffle=a.variant.endswith("_shuffle"))
        rv = torch.stack([rf[n].float() for n in names])
    torch.manual_seed(a.seed)
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    model = K7HyperRegressor(x.shape[1], sv.shape[1], hidden=a.hidden,
                             rank=a.rank, use_relation=use_relation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    # Transfer fold tensors once when GPU memory allows; otherwise stream
    # pinned CPU batches rather than silently falling back to CPU training.
    raw = (x, context, sv) + ((rv,) if use_relation else ())
    total_bytes = sum(t.numel()*t.element_size() for t in raw)
    reserve_bytes = max(2*1024**3, total_bytes*2)
    can_reside = device.type == "cuda" and (
        torch.cuda.mem_get_info()[0] > total_bytes + reserve_bytes)
    if can_reside:
        try:
            all_features = tuple(t.to(device) for t in raw)
            train_indices = torch.as_tensor(train_idx, dtype=torch.long, device=device)
            train_labels = target.to(device)
            print(f"[k7] device={torch.cuda.get_device_name(0)} "
                  f"resident=GPU feature_memory={total_bytes/1024**3:.3f}GiB "
                  f"train_rows={len(train_idx)}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            can_reside = False
    if not can_reside:
        all_features = raw
        train_indices = torch.as_tensor(train_idx, dtype=torch.long)
        train_labels = target
        print(f"[k7] device={device} resident=CPU "
              f"streaming_batches_to_GPU={device.type=='cuda'} "
              f"feature_memory={total_bytes/1024**3:.3f}GiB", flush=True)
    generator = torch.Generator(device="cpu").manual_seed(a.seed)
    outer = tqdm(range(a.epochs), desc=f"K7 {a.variant} {task[:17]}", leave=False,
                 dynamic_ncols=True)
    for epoch in outer:
        model.train()
        total = 0.0
        count = 0
        order = torch.randperm(len(train_idx), generator=generator)
        if can_reside:
            order = order.to(device)
        for start in range(0, len(train_idx), a.batch_size):
            sample_ids = order[start:start+a.batch_size]
            global_ids = train_indices[sample_ids]
            features = [t[global_ids] for t in all_features]
            labels = train_labels[sample_ids]
            if not can_reside:
                features = [t.to(device, non_blocking=True) for t in features]
                labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*features)
            loss = F.mse_loss(prediction, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            count += 1
        outer.set_postfix(loss=f"{total/max(count,1):.3f}")
    model.eval()
    records = []
    for partition in ("SCUI", "UCSI", "UCUI"):
        idx = splits[partition]
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(idx), a.batch_size):
                j = idx[start:start+a.batch_size]
                ids = torch.as_tensor(j, dtype=torch.long,
                                      device=device if can_reside else "cpu")
                features = [t[ids] for t in all_features]
                if not can_reside:
                    features = [t.to(device) for t in features]
                chunks.append(model(*features).cpu())
        standardized = torch.cat(chunks).numpy()
        mean = np.asarray([
            unseen[0] if names[i] == task else stats[names[i]][0]
            for i in idx], dtype=float)
        std = np.asarray([
            unseen[1] if names[i] == task else stats[names[i]][1]
            for i in idx], dtype=float)
        prediction = mean + std * standardized
        truth = y[idx].numpy()
        summary = scores(prediction, truth)
        baseline = float(np.mean((mean - truth) ** 2))
        records.append({
            "variant": a.variant, "heldout_task": task, "partition": partition,
            **summary, "source_mean_mse": baseline,
            "relative_mse_to_source_mean": summary["mse_log1p"]/max(baseline, 1e-12),
            **diagnostic(prediction, truth),
            "seed": a.seed, "epochs": a.epochs,
            "n_train": len(train_idx),
        })
        print(f"[k7] {task:<29} {a.variant:<21} {partition:<4} "
              f"MSE={summary['mse_log1p']:.3f} "
              f"Pearson={summary['pearson']:.3f} "
              f"Ratio={records[-1]['relative_mse_to_source_mean']:.3f}", flush=True)
    return records, {"model": model.cpu().state_dict(),
                     "heldout_task": task, "heldout_cities": held_cities,
                     "source_stats": stats, "inferred_unseen_stats": unseen}


def csv_write(path, rows):
    if not rows:
        raise ValueError("Empty metrics output")
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--source-csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--image-column")
    p.add_argument("--feature-dir", type=Path, required=True)
    p.add_argument("--remoteclip-model", default="ViT-B-32",
                   choices=("ViT-B-32", "ViT-L-14"))
    p.add_argument("--text-cache-dir", type=Path, default=Path("runs/ktrans_text"))
    p.add_argument("--relation-cache-dir", type=Path, default=Path("runs/ktrans_text"))
    p.add_argument("--text-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--city-fraction", type=float, default=0.25)
    p.add_argument("--scale-temperature", type=float, default=0.2)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.rank < 1 or not 0 < a.city_fraction < 0.5:
        p.error("Invalid Z1 experiment parameters")
    a.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v
              for k, v in vars(a).items() if k != "output"}
    config.update(version=VERSION, warning="Source pseudo-task DEV only. Oracle diagnostics not zero-shot. Fixed short semantics for numeric scale.")
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["fingerprint"] = fingerprint
    config_path = a.output/"config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Z1 resume config differs; use a fresh output directory")
    else:
        config_path.write_text(json.dumps(config, indent=2))
    embeddings = get_semantics(a)
    relations = (relation_embeddings(a.relation_cache_dir, a.text_model)
                 if a.variant != "k7_e1" else None)
    frame, x, context, y, names = load_data(a)
    print(f"[k7] data ready rows={len(frame)} visual_dim={x.shape[1]} "
          f"GPU={torch.cuda.is_available() and not a.cpu}", flush=True)
    tasks = sorted(SOURCE_TASKS)
    completed = []
    progress = tqdm(tasks, desc=f"K7 {a.variant} seed={a.seed}",
                    dynamic_ncols=True)
    for task in progress:
        row_file = a.output/(task+"_metrics.json")
        if row_file.exists():
            rows = json.loads(row_file.read_text())
            if len(rows) != 3 or {r["partition"] for r in rows} != {"SCUI", "UCSI", "UCUI"}:
                raise ValueError("Corrupt completed fold: "+str(row_file))
            completed += rows
            progress.set_postfix(done=len(completed)//3, skipped=task[:7])
            continue
        rows, checkpoint = fold_run(a, task, frame, x, context, y, names,
                                    embeddings, relations)
        torch.save(checkpoint, a.output/(task+"_model.pt"))
        temp = row_file.with_suffix(".json.tmp")
        temp.write_text(json.dumps(rows, indent=2))
        temp.replace(row_file)
        completed += rows
        progress.set_postfix(done=len(completed)//3)
    csv_write(a.output/"fold_metrics.csv", completed)
    macro = []
    for partition in ("SCUI", "UCSI", "UCUI"):
        sub = [r for r in completed if r["partition"] == partition]
        macro.append({
            "variant": a.variant, "seed": a.seed, "partition": partition,
            "n_tasks": len(sub),
            "mean_mse_log1p": float(np.mean([r["mse_log1p"] for r in sub])),
            "mean_pearson": float(np.nanmean([r["pearson"] for r in sub])),
            "mean_relative_mse": float(np.mean([
                r["relative_mse_to_source_mean"] for r in sub])),
        })
    csv_write(a.output/"macro_metrics.csv", macro)
    print(f"[k7] COMPLETE variant={a.variant} seed={a.seed} {a.output}",
          flush=True)


if __name__ == "__main__":
    main()
