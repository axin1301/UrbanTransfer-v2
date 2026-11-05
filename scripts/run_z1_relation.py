"""Unattended Z1 semantics/relation experiments with resumable source-only folds.

Run one condition/seed per command. No true target CSV is ever loaded.
Stores each finished fold before continuing; fixed budgets, no UCUI tuning.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

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
from urban_transfer.z1_relation import RelationRegressor, fold_relations, relation_embeddings

VARIANTS = (
    "short_b1", "structured_b1", "structured_b3",
    "relation_b1", "relation_shuffle_b1",
    "relation_b3", "relation_shuffle_b3",
)
VERSION = "z1-relation-overnight-v1"


def get_semantics(a):
    profile = "short" if a.variant == "short_b1" else "structured"
    return qwen_task_vectors(root=a.text_cache_dir, model_id=a.text_model,
                             profile=profile, only_cached=True)


def load_data(a):
    store = FeatureStore(a.feature_dir, encoder_id="RemoteCLIP-" + a.remoteclip_model)
    ds = CachedFeatureDataset(a.source_csv, a.image_root, store,
                              partition="eval", image_column=a.image_column,
                              seed=a.seed)
    if set(ds.df.indicator_column) != set(SOURCE_TASKS):
        raise ValueError("Source CSV must contain exactly six source indicators")
    if ds.df.city.astype(str).nunique() < 3:
        raise ValueError("At least three source cities required")
    x, context, y, names = tensorize(ds, {})
    return ds.df, x, context, y, names


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
    train_mean = torch.tensor([stats[names[i]][0] for i in train_idx])
    train_std = torch.tensor([stats[names[i]][1] for i in train_idx])
    target = (y[train_idx] - train_mean) / train_std
    sv = torch.stack([semantics[n] for n in names])
    use_relation = a.variant.startswith("relation_")
    rv = None
    if use_relation:
        rf = fold_relations(relation, task, SOURCE_TASKS,
                            shuffle="shuffle" in a.variant)
        rv = torch.stack([rf[n] for n in names])
        dataset = TensorDataset(x[train_idx], context[train_idx],
                                sv[train_idx], rv[train_idx], target)
    else:
        dataset = TensorDataset(x[train_idx], context[train_idx],
                                sv[train_idx], target)
    mode = "b3" if a.variant.endswith("b3") else "b1"
    torch.manual_seed(a.seed)
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    model = (RelationRegressor(x.shape[1], sv.shape[1], mode, a.hidden, a.rank)
             if use_relation else
             Z1Regressor(x.shape[1], sv.shape[1], mode, a.hidden, a.rank)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed))
    outer = tqdm(range(a.epochs), desc=f"{a.variant} {task[:17]}", leave=False,
                 dynamic_ncols=True)
    for epoch in outer:
        model.train()
        total = 0.0
        for samples in loader:
            *features, labels = [z.to(device) for z in samples]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*features)
            loss = F.mse_loss(prediction, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach().cpu())
        outer.set_postfix(loss=f"{total / max(len(loader),1):.3f}")
    model.eval()
    records = []
    for partition in ("SCUI", "UCSI", "UCUI"):
        idx = splits[partition]
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(idx), a.batch_size):
                j = idx[start:start+a.batch_size]
                features = [x[j].to(device), context[j].to(device), sv[j].to(device)]
                if use_relation:
                    features.append(rv[j].to(device))
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
        print(f"[z1-rel] {task:<29} {a.variant:<21} {partition:<4} "
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
    p.add_argument("--rank", type=int, default=16)
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
    config.update(version=VERSION, warning="Source pseudo-task DEV only. Oracle diagnostics not zero-shot.")
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
                 if a.variant.startswith("relation_") else None)
    frame, x, context, y, names = load_data(a)
    tasks = sorted(SOURCE_TASKS)
    completed = []
    progress = tqdm(tasks, desc=f"Z1 {a.variant} seed={a.seed}",
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
    print(f"[z1-rel] COMPLETE variant={a.variant} seed={a.seed} {a.output}",
          flush=True)


if __name__ == "__main__":
    main()
