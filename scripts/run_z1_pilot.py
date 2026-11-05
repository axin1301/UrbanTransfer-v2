"""Z1 source-only leave-one-indicator/city regression pilot; no VLM generation.

B1/B2/B3 are trained from identical frozen RemoteCLIP and text features.
All pseudo-target labels are excluded from optimization and calibration.
No development UCUI labels are used for checkpoint selection: fixed epochs.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from urban_transfer.data_m1 import task_vector
from urban_transfer.ktrans_text import qwen_task_vectors
from urban_transfer.m3_features import CachedFeatureDataset, FeatureStore
from urban_transfer.protocol import SOURCE_TASKS, TASK_DESCRIPTIONS
from urban_transfer.z1_model import Z1Regressor


def fold_indices(frame, held_task, held_cities):
    """Four disjoint city x task partitions, including zero-shot UCUI."""
    task = frame.indicator_column.astype(str).eq(held_task).to_numpy()
    unseen = frame.city.astype(str).isin(held_cities).to_numpy()
    result = {
        "train": np.flatnonzero(~task & ~unseen),
        "SCUI": np.flatnonzero(task & ~unseen),
        "UCSI": np.flatnonzero(~task & unseen),
        "UCUI": np.flatnonzero(task & unseen),
    }
    if any(len(v) == 0 for v in result.values()):
        raise ValueError("Empty Z1 fold: increase city/indicator coverage")
    return result


def semantic_vectors(args):
    if args.semantic_backend == "hash":
        if not args.allow_hash_smoke:
            raise ValueError("Hash semantics are for smoke tests; pass --allow-hash-smoke")
        return {t: torch.tensor(task_vector(TASK_DESCRIPTIONS[t], 128))
                for t in SOURCE_TASKS}
    try:
        vectors = qwen_task_vectors(root=args.text_cache_dir, model_id=args.text_model,
                                    profile=args.text_profile, only_cached=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            str(exc) + "; precompute Qwen text embeddings before running Z1") from exc
    missing = set(SOURCE_TASKS) - set(vectors)
    if missing:
        raise ValueError("Frozen text embeddings missing source tasks: " + repr(missing))
    return {t: F.normalize(vectors[t].float().flatten(), dim=0) for t in SOURCE_TASKS}


def calibrated_moments(train_frame, semantics, held_task, temperature=0.2):
    """Training-only per-source moments; unseen task moments from text similarity."""
    if temperature <= 0:
        raise ValueError("Positive scale routing temperature required")
    available = sorted(set(train_frame.indicator_column))
    if held_task in available:
        raise ValueError("Held-out task leaked into source scale fitting")
    stats = {}
    for task in available:
        values = train_frame.loc[
            train_frame.indicator_column == task, "indicator_value_log1p"
        ].to_numpy(dtype=np.float64)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("Insufficient source labels for " + task)
        stats[task] = (float(values.mean()), max(float(values.std()), 1e-3))
    scores = torch.tensor([
        float(F.cosine_similarity(semantics[held_task], semantics[t], dim=0))
        for t in available
    ])
    weights = torch.softmax(scores / temperature, dim=0).tolist()
    unseen_mean = sum(w * stats[t][0] for w, t in zip(weights, available))
    unseen_std = sum(w * stats[t][1] for w, t in zip(weights, available))
    return stats, (unseen_mean, unseen_std)


def tensorize(dataset, semantics):
    """Read frozen cached visual features once; never decode/encode images here."""
    images, contexts, y, ids = [], [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        images.append(item["image"].float())
        contexts.append(item["context"].float())
        y.append(float(item["target"]))
        ids.append(str(item["task_name"]))
    return (torch.stack(images), torch.stack(contexts),
            torch.tensor(y, dtype=torch.float32), ids)


def scores(pred, truth):
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("Nonfinite Z1 prediction/label")
    mse = float(np.mean((pred - truth) ** 2))
    mae = float(np.mean(np.abs(pred - truth)))
    if len(pred) < 2 or min(float(np.std(pred)), float(np.std(truth))) <= 1e-10:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(pred, truth)[0, 1])
    return {"n_rows": len(pred), "mse_log1p": mse,
            "mae_log1p": mae, "pearson": pearson}


def execute(args):
    if args.epochs < 1 or args.batch_size < 1 or not (0 < args.city_fraction < 0.5):
        raise ValueError("Invalid Z1 hyperparameters")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Z1 output must be new/empty to protect prior results: " + str(args.output))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    embeddings = semantic_vectors(args)
    store = FeatureStore(args.feature_dir, encoder_id="RemoteCLIP-" + args.remoteclip_model)
    ds = CachedFeatureDataset(args.source_csv, args.image_root, store,
                              partition="eval", image_column=args.image_column,
                              limit=args.limit, seed=args.seed)
    if not set(ds.df.indicator_column).issubset(SOURCE_TASKS):
        raise ValueError("Z1 pilot only accepts known source indicators")
    tasks = sorted(set(ds.df.indicator_column))
    if set(tasks) != set(SOURCE_TASKS):
        raise ValueError("Z1 pilot needs all six source tasks")
    cities = sorted(ds.df.city.astype(str).unique())
    if len(cities) < 3:
        raise ValueError("Z1 needs at least three source cities")
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(cities)
    n_held = max(1, min(len(cities)-2, int(round(len(cities)*args.city_fraction))))
    held_cities = sorted(permutation[:n_held].tolist())
    images, contexts, labels, names = tensorize(ds, embeddings)
    sem = torch.stack([embeddings[t] for t in names])
    modes = tuple(args.modes)
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config.update({"version": "z1-source-loto-v1", "source_tasks": tasks,
                   "held_cities": held_cities, "n_rows": len(ds),
                   "visual_dim": int(images.shape[1]),
                   "semantic_dim": int(sem.shape[1]),
                   "target_policy": "held-out task and city excluded from training; source-only statistics",
                   "selection_policy": "fixed epochs and hyperparameters; no UCUI model selection",
                   "test_warning": "DEV pseudo-task results, NOT independent four-target zero-shot evaluation"})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    records = []
    for held_task in tasks:
        split = fold_indices(ds.df, held_task, held_cities)
        train_frame = ds.df.iloc[split["train"]]
        stats, unseen_stats = calibrated_moments(train_frame, embeddings, held_task,
                                                 args.scale_temperature)
        # Source-only per-indicator normalization, shared semantic regression head.
        mean = torch.tensor([stats[names[i]][0] for i in split["train"]])
        std = torch.tensor([stats[names[i]][1] for i in split["train"]])
        train_y = (labels[split["train"]] - mean) / std
        train_dataset = TensorDataset(images[split["train"]], contexts[split["train"]],
                                      sem[split["train"]], train_y)
        for mode in modes:
            torch.manual_seed(args.seed)  # identical fold/mode initialization seed
            model = Z1Regressor(images.shape[1], sem.shape[1],
                                mode=mode, hidden=args.hidden, rank=args.rank).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, generator=torch.Generator().manual_seed(args.seed))
            for epoch in range(args.epochs):
                model.train()
                losses = []
                for xv, xc, xt, yt in loader:
                    xv, xc, xt, yt = (v.to(device) for v in (xv, xc, xt, yt))
                    opt.zero_grad()
                    loss = F.mse_loss(model(xv, xc, xt), yt)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                    losses.append(float(loss.detach().cpu()))
                if epoch == 0 or (epoch+1) % args.log_every == 0 or epoch+1 == args.epochs:
                    print(f"[z1] task={held_task} mode={mode} epoch={epoch+1}/{args.epochs} train_loss={np.mean(losses):.4f}", flush=True)
            model.eval()
            for partition in ("SCUI", "UCSI", "UCUI"):
                indices = split[partition]
                with torch.inference_mode():
                    zs = []
                    for start in range(0, len(indices), args.batch_size):
                        j = indices[start:start+args.batch_size]
                        zs.append(model(images[j].to(device), contexts[j].to(device),
                                        sem[j].to(device)).cpu())
                    standardized = torch.cat(zs).numpy()
                means = np.array([unseen_stats[0] if names[i] == held_task else stats[names[i]][0]
                                  for i in indices])
                stds = np.array([unseen_stats[1] if names[i] == held_task else stats[names[i]][1]
                                 for i in indices])
                predicted = means + stds * standardized
                summary = scores(predicted, labels[indices].numpy())
                baseline = scores(means, labels[indices].numpy())
                row = {"mode": mode, "heldout_task": held_task, "partition": partition,
                       **summary, "source_mean_mse": baseline["mse_log1p"],
                       "relative_mse_to_source_mean": summary["mse_log1p"] /
                           max(baseline["mse_log1p"], 1e-12),
                       "epoch": args.epochs, "n_train": len(split["train"])}
                records.append(row)
                print(f"[z1] {mode:<3} {held_task:<30} {partition:<4} "
                      f"mse={summary['mse_log1p']:.4f} pearson={summary['pearson']:.3f} "
                      f"ratio={row['relative_mse_to_source_mean']:.3f}", flush=True)
            # Store model checkpoint per fold so results remain inspectable.
            ckpt = args.output / ("model_" + mode + "_" + held_task + ".pt")
            torch.save({"model": model.cpu().state_dict(), "config": config,
                        "heldout_task": held_task, "held_cities": held_cities,
                        "source_stats": stats, "unseen_stats": unseen_stats}, ckpt)
    out_csv = args.output / "fold_metrics.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    macro = []
    for mode in modes:
        for partition in ("SCUI", "UCSI", "UCUI"):
            sub = [r for r in records if r["mode"] == mode and r["partition"] == partition]
            macro.append({"mode": mode, "partition": partition, "n_tasks": len(sub),
                          "mean_mse_log1p": float(np.mean([r["mse_log1p"] for r in sub])),
                          "mean_pearson": float(np.nanmean([r["pearson"] for r in sub])),
                          "mean_relative_mse": float(np.mean(
                              [r["relative_mse_to_source_mean"] for r in sub]))})
    with (args.output / "macro_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(macro[0]))
        w.writeheader()
        w.writerows(macro)
    print(f"[z1] complete: {len(records)} fold rows saved in {args.output}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv", type=Path, required=True,
                   help="CSV of source-task labeled rows across source cities")
    p.add_argument("--image-root", type=Path)
    p.add_argument("--image-column")
    p.add_argument("--feature-dir", type=Path, required=True,
                   help="Existing frozen RemoteCLIP SQLite cache directory")
    p.add_argument("--remoteclip-model", choices=("ViT-B-32", "ViT-L-14"), default="ViT-B-32")
    p.add_argument("--text-cache-dir", type=Path, default=Path("runs/ktrans_text"))
    p.add_argument("--text-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--text-profile", default="short")
    p.add_argument("--semantic-backend", choices=("qwen", "hash"), default="qwen")
    p.add_argument("--allow-hash-smoke", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modes", nargs="+", choices=("b1", "b2", "b3"),
                   default=("b1", "b2", "b3"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--city-fraction", type=float, default=0.25)
    p.add_argument("--scale-temperature", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.hidden < 8 or args.rank < 1 or args.log_every < 1:
        p.error("Invalid architecture/log settings")
    if not args.image_root and not args.image_column:
        p.error("Provide --image-root or --image-column")
    execute(args)


if __name__ == "__main__":
    main()
