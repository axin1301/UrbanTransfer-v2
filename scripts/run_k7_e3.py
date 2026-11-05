"""K7 E3: single-seed, two-condition, source-only reward-calibrated routing.

Reuses K7-E1 fold checkpoints; trains one inner-city reward model per held-out
source task. Compare correct vs shuffled directed relation knowledge with
identical learned transfer rewards and identical base checkpoints.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm.auto import tqdm

from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_hyper import K7HyperRegressor
from urban_transfer.k7_utility import (
    fit_inner, inner_city_split, measured_utilities,
    predict_adapter, predicted_weights)
from urban_transfer.ktrans_text import qwen_task_vectors
from urban_transfer.protocol import SOURCE_TASKS
from urban_transfer.z1_relation import relation_embeddings
from scripts.run_z1_pilot import fold_indices, calibrated_moments, scores
from scripts.run_z1_relation import diagnostic

VERSION = "k7-e3-utility-v1"
VARIANTS = ("k7_e3", "k7_e3_shuffle")


def atomic_csv(path, rows):
    if not rows:
        raise ValueError("Missing result rows")
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    temp.replace(path)


def evaluate(a, task, variant, split, model, data, semantics, relations, rewards,
             source_stats, unseen_stats, device):
    x, ctx, y, names = data
    sources = sorted(source_stats)
    weights, utilities = predicted_weights(
        task, sources, rewards, semantics, relations,
        shuffle=(variant == "k7_e3_shuffle"), alpha=a.ridge_alpha)
    results = []
    # Only the held-indicator partitions are defined for this router:
    # UCSI spans five known indicators and would require five routers.
    for partition in ("SCUI", "UCUI"):
        ids = split[partition]
        z = weights[0] * predict_adapter(model, x, ctx, ids, semantics[task],
                                         a.batch_size, device)
        for j, source in enumerate(sources):
            z += weights[j+1] * predict_adapter(
                model, x, ctx, ids, semantics[source], a.batch_size, device)
        estimate = unseen_stats[0] + unseen_stats[1] * z
        observed = y[ids].numpy()
        summary = scores(estimate, observed)
        baseline = float(np.mean((unseen_stats[0]-observed)**2))
        row = {
            "variant": variant, "seed": a.seed, "heldout_task": task,
            "partition": partition, **summary, "source_mean_mse": baseline,
            "relative_mse_to_source_mean": summary["mse_log1p"]/max(baseline, 1e-12),
            **diagnostic(estimate, observed),
            "direct_adapter_weight": float(weights[0]),
            "max_predicted_transfer_utility": float(np.max(utilities)),
            "n_reward_pairs": len(rewards),
        }
        results.append(row)
        print(f"[k7-e3] {variant:<14} {task:<29} {partition} "
              f"MSE={summary['mse_log1p']:.3f} Pearson={summary['pearson']:.3f} "
              f"Ratio={row['relative_mse_to_source_mean']:.3f} "
              f"direct_w={weights[0]:.3f}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--feature-dir", type=Path, default=Path("runs/m3_features"))
    p.add_argument("--tensor-cache", type=Path,
                   default=Path("runs/k7/e1_e2_v1/_source_tensors.pt"))
    p.add_argument("--e1-root", type=Path,
                   default=Path("runs/k7/e1_e2_v1/k7_e1_seed42"))
    p.add_argument("--output", type=Path, default=Path("runs/k7/e3_seed42"))
    p.add_argument("--remoteclip-model", default="ViT-B-32")
    p.add_argument("--text-cache-dir", type=Path, default=Path("runs/ktrans_text"))
    p.add_argument("--relation-cache-dir", type=Path, default=Path("runs/ktrans_text"))
    p.add_argument("--text-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--city-fraction", type=float, default=0.25)
    p.add_argument("--inner-city-fraction", type=float, default=0.2)
    p.add_argument("--reward-max-rows", type=int, default=2048)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--scale-temperature", type=float, default=0.2)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("No ROCm/CUDA GPU visible; K7 E3 refuses silent CPU training")
    if a.seed != 42:
        raise ValueError("Current K7 E3 supports seed42 E1 checkpoints only")
    if a.epochs < 1 or a.batch_size < 1 or a.reward_max_rows < 2:
        p.error("Invalid training budget")
    if not a.tensor_cache.is_file() or not a.e1_root.is_dir():
        raise FileNotFoundError("Need finished K7 E1 seed42 and shared tensor cache")
    a.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v
              for k,v in vars(a).items() if k != "output"}
    config["version"] = VERSION
    config["fingerprint"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = a.output/"config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Resume config differs; use a new output directory")
    else:
        config_path.write_text(json.dumps(config, indent=2))
    frame, x, ctx, y, names = read_tensor_cache(
        a.tensor_cache, a.source_csv, a.image_root, a.feature_dir, a.remoteclip_model)
    semantics = qwen_task_vectors(
        root=a.text_cache_dir, model_id=a.text_model,
        profile="short", only_cached=True)
    relations = relation_embeddings(a.relation_cache_dir, a.text_model, only_cached=True)
    if set(frame.indicator_column) != set(SOURCE_TASKS):
        raise ValueError("Not a six-source-only development dataset")
    device = torch.device("cuda")
    x, ctx = x.to(device), ctx.to(device)
    data = (x, ctx, y, names)
    print(f"[k7-e3] GPU={torch.cuda.get_device_name(0)} "
          f"rows={len(frame)} feature_GiB={x.numel()*x.element_size()/1024**3:.3f}",
          flush=True)
    all_rows = []
    for task in tqdm(sorted(SOURCE_TASKS), desc="K7 E3 6 inner-city folds",
                     dynamic_ncols=True):
        result = a.output/(task+"_metrics.json")
        if result.exists():
            rows = json.loads(result.read_text())
            if len(rows) != 4 or {r["variant"] for r in rows} != set(VARIANTS):
                raise ValueError("Corrupt saved K7 E3 fold " + str(result))
            all_rows.extend(rows)
            continue
        cities = sorted(frame.city.astype(str).unique())
        perm = np.random.default_rng(a.seed).permutation(cities)
        count = max(1, min(len(cities)-2, round(len(cities)*a.city_fraction)))
        held_cities = sorted(perm[:count].tolist())
        split = fold_indices(frame, task, held_cities)
        checkpoint = a.e1_root/(task+"_model.pt")
        if not checkpoint.is_file():
            raise FileNotFoundError("Missing trained E1 fold " + str(checkpoint))
        stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if sorted(stored["heldout_cities"]) != held_cities:
            raise ValueError("K7 E1 checkpoint uses different held-out cities")
        full = K7HyperRegressor(
            x.shape[1], len(semantics[task]), a.hidden, a.rank, False).to(device)
        full.load_state_dict(stored["model"], strict=True)
        full.eval()
        stats, unseen = calibrated_moments(
            frame.iloc[split["train"]], semantics, task, a.scale_temperature)
        inner_train, inner_val = inner_city_split(
            frame, split["train"], a.seed, a.inner_city_fraction)
        print(f"[k7-e3] task={task} inner_train={len(inner_train)} "
              f"inner_reward_val={len(inner_val)}", flush=True)
        inner, moments = fit_inner(
            a, frame, x, ctx, y, names, inner_train, semantics, device)
        rewards = measured_utilities(
            a, inner, frame, x, ctx, y, names, inner_val,
            moments, semantics, device)
        del inner
        print(f"[k7-e3] task={task} measured_pairs={len(rewards)} "
              f"positive_rewards={sum(v>0 for v in rewards.values())}",
              flush=True)
        rows = []
        for variant in VARIANTS:
            rows.extend(evaluate(a, task, variant, split, full, data,
                                 semantics, relations, rewards, stats, unseen, device))
        del full
        temp = result.with_suffix(".json.tmp")
        temp.write_text(json.dumps(rows, indent=2))
        temp.replace(result)
        all_rows.extend(rows)
    atomic_csv(a.output/"fold_metrics.csv", all_rows)
    macros = []
    for variant in VARIANTS:
        for partition in ("SCUI", "UCUI"):
            subset = [r for r in all_rows if r["variant"] == variant
                      and r["partition"] == partition]
            macros.append({
                "variant": variant, "seed": a.seed, "partition": partition,
                "n_tasks": len(subset),
                "mean_mse_log1p": float(np.mean([r["mse_log1p"] for r in subset])),
                "mean_pearson": float(np.nanmean([r["pearson"] for r in subset])),
                "mean_relative_mse": float(np.mean([
                    r["relative_mse_to_source_mean"] for r in subset])),
            })
    atomic_csv(a.output/"macro_metrics.csv", macros)
    print("Model              UCUI MSE   Pearson   Ratio", flush=True)
    for row in macros:
        if row["partition"] == "UCUI":
            print(f"{row['variant']:<19}{row['mean_mse_log1p']:>8.3f} "
                  f"{row['mean_pearson']:>9.3f} "
                  f"{row['mean_relative_mse']:>7.3f}", flush=True)
    print(f"[k7-e3] COMPLETE {a.output}", flush=True)


if __name__ == "__main__":
    main()
