"""M2 counterfactual audit: concrete directed mechanism text vs mismatched/generic text.

INFERENCE ONLY: no model training, labels used only for scoring. Keep source
identity, source experts, target semantics, scale and city splits unchanged.
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
import torch.nn.functional as F
from tqdm.auto import tqdm

from scripts.run_k7_m1 import card_path
from scripts.run_k7_m2 import build_tokens
from scripts.run_z1_pilot import scores
from urban_transfer.k7_directed_fusion import DirectedEdgeFusionRegressor
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS, HELDOUT_TASK
from urban_transfer.ktrans_text import last_token_pool, qwen_task_vectors
from urban_transfer.task_profiles import PROFILES

CONDITIONS = ("correct", "swapped", "generic")


def generic_relation_text(source, target):
    # Source and target identities/definitions remain explicit; only the
    # pair-specific spatial/transfer mechanisms are removed.
    src_definition, _, _ = PROFILES[source]
    tgt_definition, _, _ = PROFILES[target]
    return (
        "Directed transfer of urban indicator prediction knowledge.\n"
        f"Source: {source}. Definition: {src_definition}\n"
        f"Target: {target}. Definition: {tgt_definition}\n"
        "Potential shared visual evidence: unspecified.\n"
        "Measurement link: unspecified.\n"
        "Transferable representation: unspecified.\n"
        "Do NOT transfer: source output units and target output scale are not interchangeable.\n"
        "Evidence limitation: no pair-specific mechanism knowledge is provided.\n"
        "Do not assert causal relations, exact target ranges, target means, "
        "target standard deviations, or known empirical transfer accuracy."
    )


def generic_cache_path(root, model_id):
    texts = {source: generic_relation_text(source, HELDOUT_TASK)
             for source in TRAIN_TASKS}
    payload = json.dumps({"texts": texts, "model": model_id, "v": 1,
                          "pool": "last-token", "max_length": 1024},
                         sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:20]
    return Path(root)/f"k7_m2_generic_{digest}.pt"


def generic_embeddings(root, model_id, dimension):
    path = generic_cache_path(root, model_id)
    if path.exists():
        vectors = torch.load(path, map_location="cpu", weights_only=True)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required to encode two generic Qwen cards")
        from transformers import AutoModel, AutoTokenizer
        print("[m2-audit] encoding 2 generic relation cards using frozen Qwen...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to("cuda").eval()
        texts = [generic_relation_text(source, HELDOUT_TASK) for source in TRAIN_TASKS]
        with torch.inference_mode():
            batch = tokenizer(texts, padding=True, truncation=True,
                              max_length=1024, return_tensors="pt").to("cuda")
            hidden = model(**batch).last_hidden_state
            vectors_batch = F.normalize(
                last_token_pool(hidden, batch["attention_mask"]).float(), dim=-1).cpu()
        vectors = {source: vectors_batch[i].clone()
                   for i, source in enumerate(TRAIN_TASKS)}
        del model
        torch.cuda.empty_cache()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vectors, path)
        print(f"[m2-audit] generic embeddings saved: {path}", flush=True)
    if set(vectors) != set(TRAIN_TASKS):
        raise ValueError("Generic card cache source mismatch")
    for source in TRAIN_TASKS:
        vec = vectors[source]
        if vec.numel() != dimension or not torch.isfinite(vec).all():
            raise ValueError("Generic card dimension/finite mismatch")
    return {source: F.normalize(vectors[source].float().flatten(), dim=0)
            for source in TRAIN_TASKS}


def relation_conditions(correct, generic):
    """Swapping the two vectors but NOT their source experts/identities."""
    if correct.shape != generic.shape or correct.ndim != 2 or len(correct) != 2:
        raise ValueError("Expected two matched directed relation vectors")
    return {
        "correct": correct,
        "swapped": correct.flip(0).clone(),
        "generic": generic,
    }


def predict(model, x, ctx, indices, target_sem, source_sem,
            edge_rel, flags, batch_size, device):
    outputs, attentions = [], []
    with torch.inference_mode():
        for start in tqdm(range(0, len(indices), batch_size),
                          leave=False, dynamic_ncols=True, desc="M2 audit inference"):
            ix = torch.as_tensor(indices[start:start+batch_size], dtype=torch.long)
            b = len(ix)
            result, weight = model(
                x[ix].to(device), ctx[ix].to(device),
                target_sem.expand(b,-1), source_sem,
                edge_rel.expand(b,-1,-1), flags.expand(b,-1),
                return_attention=True)
            outputs.append(result.cpu())
            attentions.append(weight.cpu())
    return (torch.cat(outputs).numpy(),
            torch.cat(attentions).mean(0).numpy())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path,
                   default=Path("runs/k7/m2_directed_fusion_seed42"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("batch-size must be positive")
    run_dir = a.run_dir
    config_file, checkpoint_file = run_dir/"config.json", run_dir/"model.pt"
    if not config_file.is_file() or not checkpoint_file.is_file():
        raise FileNotFoundError("Completed M2 run (config.json + model.pt) required")
    config = json.loads(config_file.read_text())
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if (tuple(config["source_tasks"]) != TRAIN_TASKS or
        config["heldout_task"] != HELDOUT_TASK or
        tuple(checkpoint["source_tasks"]) != TRAIN_TASKS or
        checkpoint["heldout_task"] != HELDOUT_TASK):
        raise ValueError("This audit requires the fixed Population+Food -> Education M2 run")
    frame, x, context, y, names = read_tensor_cache(
        Path(config["tensor_cache"]), Path(config["source_csv"]),
        Path(config["image_root"]), Path(config["feature_dir"]),
        config["remoteclip_model"])
    embeddings = qwen_task_vectors(
        root=Path(config["text_cache_dir"]), model_id=config["text_model"],
        profile="short", only_cached=True)
    path = card_path(Path(config["text_cache_dir"]), config["text_model"])
    if not path.is_file():
        raise FileNotFoundError("Missing prior M2 mechanism card embeddings: " + str(path))
    cards = torch.load(path, map_location="cpu", weights_only=True)
    pair = {(source, HELDOUT_TASK):
            F.normalize(cards[f"{source}|{HELDOUT_TASK}"].float(), dim=0)
            for source in TRAIN_TASKS}
    original, flags = build_tokens(HELDOUT_TASK, embeddings, pair)
    generic = generic_embeddings(Path(config["text_cache_dir"]),
                                 config["text_model"], original.shape[1])
    generic_tensor = torch.stack([generic[source] for source in TRAIN_TASKS])
    conditions = relation_conditions(original, generic_tensor)
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    model = DirectedEdgeFusionRegressor(
        x.shape[1], embeddings[HELDOUT_TASK].numel(),
        hidden=int(config["hidden"]), rank=int(config["rank"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    target_sem = embeddings[HELDOUT_TASK].float().to(device)
    source_sem = torch.stack([embeddings[src].float() for src in TRAIN_TASKS]).to(device)
    flags = flags.to(device)
    mean, std = (float(value) for value in checkpoint["unseen_stats"])
    unseen = frame.city.astype(str).isin(checkpoint["heldout_cities"]).to_numpy()
    target_mask = frame.indicator_column.eq(HELDOUT_TASK).to_numpy()
    print(f"[m2-audit] device={device} training=NONE fixed_checkpoint={checkpoint_file}",
          flush=True)
    rows = []
    for part, ids in (("SCUI", np.flatnonzero(target_mask & ~unseen)),
                      ("UCUI", np.flatnonzero(target_mask & unseen))):
        if len(ids) == 0:
            raise ValueError("Missing Education pseudo-validation split " + part)
        outputs = {}
        weights = {}
        for condition in CONDITIONS:
            outputs[condition], weights[condition] = predict(
                model, x, context, ids, target_sem, source_sem,
                conditions[condition].to(device), flags, a.batch_size, device)
        truth = y[ids].numpy()
        base = float(np.mean((mean-truth)**2))
        ref = mean+std*outputs["correct"]
        for condition in CONDITIONS:
            pred = mean+std*outputs[condition]
            metrics = scores(pred, truth)
            rows.append({
                "partition": part, "condition": condition, "n": len(ids),
                "mse_log1p": metrics["mse_log1p"],
                "pearson": metrics["pearson"],
                "ratio": metrics["mse_log1p"]/max(base, 1e-12),
                "delta_mse_vs_correct":
                    metrics["mse_log1p"] - float(np.mean((ref-truth)**2)),
                "mean_abs_prediction_change": float(np.mean(np.abs(pred-ref))),
                "population_attention": float(weights[condition][0]),
                "food_attention": float(weights[condition][1]),
            })
    output = a.output or run_dir/"relation_text_counterfactual.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(output)
    print(f"{'Split':5} {'Text':9} {'MSE':>7} {'Pearson':>8} {'Ratio':>7} "
          f"{'ΔMSE':>8} {'|ΔPred|':>8}", flush=True)
    for row in rows:
        print(f"{row['partition']:5} {row['condition']:9} "
              f"{row['mse_log1p']:7.3f} {row['pearson']:8.3f} "
              f"{row['ratio']:7.3f} {row['delta_mse_vs_correct']:8.3f} "
              f"{row['mean_abs_prediction_change']:8.3f}", flush=True)
    print("[m2-audit] Counterfactual inference is exploratory: generic/swapped "
          "tokens are not trained conditions and differences alone do not prove "
          "semantic understanding. No model was refit.", flush=True)
    print(f"[m2-audit] Saved {output}", flush=True)


if __name__ == "__main__":
    main()
