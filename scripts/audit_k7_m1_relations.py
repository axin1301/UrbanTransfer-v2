"""Inference-only relation dependence audit for the completed K7-M1 checkpoint.

No optimizer, backpropagation, Qwen inference, or source-label fitting.
Uses the identical saved unseen-task output scale for all interventions.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from scripts.run_k7_m1 import card_path, semantic_inputs
from scripts.run_z1_pilot import scores
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_hyper import K7HyperRegressor
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS, HELDOUT_TASK
from urban_transfer.ktrans_text import qwen_task_vectors

CASES = ("correct", "zero_relation", "source_training_relation", "swap_edges")
SPLITS = ("SCUI", "UCUI")


def interventions(embeddings, cards):
    """Swapping inputs to a mean must leave the resulting target vector unchanged."""
    vectors = [cards[src, HELDOUT_TASK].float() for src in TRAIN_TASKS]
    correct = F.normalize(torch.stack(vectors).mean(0), dim=0)
    swapped = F.normalize(torch.stack(vectors[::-1]).mean(0), dim=0)
    train_rel = [
        semantic_inputs(src, embeddings, cards)[1].float() for src in TRAIN_TASKS
    ]
    substitute = F.normalize(torch.stack(train_rel).mean(0), dim=0)
    return {
        "correct": correct,
        "zero_relation": torch.zeros_like(correct),
        "source_training_relation": substitute,
        "swap_edges": swapped,
    }


def prediction(model, x, context, ids, semantic, relation, batch, device):
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch):
            j = torch.as_tensor(ids[start:start+batch], dtype=torch.long)
            k = len(j)
            z = model(
                x[j].to(device), context[j].to(device),
                semantic.expand(k, -1), relation.expand(k, -1))
            chunks.append(z.cpu())
    return torch.cat(chunks).numpy()


def evaluate_arrays(original, candidate, truth, mean, std):
    pred = mean + std * candidate
    ref = mean + std * original
    metric = scores(pred, truth)
    baseline = float(np.mean((mean-truth)**2))
    return {
        "mse_log1p": float(metric["mse_log1p"]),
        "pearson": float(metric["pearson"]),
        "ratio": float(metric["mse_log1p"]/max(baseline, 1e-12)),
        "delta_mse_vs_correct": float(metric["mse_log1p"] -
                                      np.mean((ref-truth)**2)),
        "mean_abs_prediction_change": float(np.mean(np.abs(pred-ref))),
        "max_abs_prediction_change": float(np.max(np.abs(pred-ref))),
        "prediction_correlation_vs_correct":
            float(np.corrcoef(pred, ref)[0, 1]) if (
                np.std(pred)>1e-12 and np.std(ref)>1e-12
            ) else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path,
                   default=Path("runs/k7/m1_pop_food_to_education"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("--batch-size must be >=1")
    run = a.run_dir
    output = a.output or run/"relation_audit.csv"
    config_file, ckpt_file = run/"config.json", run/"model.pt"
    if not config_file.is_file() or not ckpt_file.is_file():
        raise FileNotFoundError("Completed K7 M1 config.json/model.pt required")
    config = json.loads(config_file.read_text())
    if tuple(config["source_tasks"]) != TRAIN_TASKS or config["heldout_task"] != HELDOUT_TASK:
        raise ValueError("Checkpoint must be Population + Food -> Education")
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    if tuple(checkpoint["source_tasks"]) != TRAIN_TASKS or checkpoint["heldout_task"] != HELDOUT_TASK:
        raise ValueError("Incompatible saved checkpoint")
    frame, x, context, y, names = read_tensor_cache(
        Path(config["tensor_cache"]), Path(config["source_csv"]),
        Path(config["image_root"]), Path(config["feature_dir"]),
        config["remoteclip_model"])
    embeddings = qwen_task_vectors(
        root=Path(config["text_cache_dir"]), model_id=config["text_model"],
        profile="short", only_cached=True)
    file = card_path(Path(config["text_cache_dir"]), config["text_model"])
    if not file.is_file():
        raise FileNotFoundError("Encoded M1 mechanism cards missing: " + str(file))
    raw = torch.load(file, map_location="cpu", weights_only=True)
    cards = {tuple(k.split("|")): F.normalize(v.float(), dim=0)
             for k, v in raw.items()}
    vectors = interventions(embeddings, cards)
    if not torch.allclose(vectors["correct"], vectors["swap_edges"], atol=1e-7):
        raise AssertionError("Edge swapping unexpectedly changed mean-pooled vector")
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[m1-audit] device={device} checkpoint={ckpt_file} "
          f"training=NONE Qwen_encoding=NONE", flush=True)
    model = K7HyperRegressor(
        x.shape[1], embeddings[HELDOUT_TASK].numel(),
        hidden=int(config["hidden"]), rank=int(config["rank"]),
        use_relation=True).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    sem = embeddings[HELDOUT_TASK].float().to(device)
    scale_mean, scale_std = [float(v) for v in checkpoint["unseen_stats"]]
    unseen = frame.city.astype(str).isin(checkpoint["heldout_cities"]).to_numpy()
    target = frame.indicator_column.eq(HELDOUT_TASK).to_numpy()
    rows = []
    for split, ids in tqdm((
            ("SCUI", np.flatnonzero(target & ~unseen)),
            ("UCUI", np.flatnonzero(target & unseen))),
            desc="M1 inference audit", dynamic_ncols=True):
        if len(ids) == 0:
            raise ValueError("Missing " + split + " Education examples")
        z = {}
        for name, rel in vectors.items():
            z[name] = prediction(model, x, context, ids, sem, rel.to(device),
                                 a.batch_size, device)
        truth = y[ids].numpy()
        for name in CASES:
            result = evaluate_arrays(z["correct"], z[name], truth,
                                     scale_mean, scale_std)
            rows.append({"partition": split, "intervention": name,
                         "n": len(ids), **result})
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(output)
    print(f"{'Split':5} {'Relation':25} {'MSE':>7} {'Pearson':>8} "
          f"{'Ratio':>7} {'ΔMSE':>8} {'|ΔPred|':>8}", flush=True)
    for r in rows:
        print(f"{r['partition']:5} {r['intervention']:25} "
              f"{r['mse_log1p']:7.3f} {r['pearson']:8.3f} "
              f"{r['ratio']:7.3f} {r['delta_mse_vs_correct']:8.3f} "
              f"{r['mean_abs_prediction_change']:8.3f}", flush=True)
    print("[m1-audit] swap_edges = correct by construction: "
          "mean pooling erases source order. ΔMSE is diagnostic only; "
          "no intervention is used for model selection.", flush=True)
    print(f"[m1-audit] saved: {output}", flush=True)


if __name__ == "__main__":
    main()
