"""K7-M1: ONE model, two supervised indicators, one unseen validation indicator.

Source labels: population_count and food_accessibility, source cities only.
Target education_accessibility labels are consulted ONLY after training.
Four audited directed mechanism cards condition a generated low-rank adapter.
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

from urban_transfer.data_m1 import stratified_sample
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_hyper import K7HyperRegressor
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS, HELDOUT_TASK, all_relation_texts
from urban_transfer.ktrans_text import qwen_task_vectors, last_token_pool
from scripts.run_z1_pilot import scores, calibrated_moments
from scripts.run_z1_relation import diagnostic


def card_path(root, model):
    texts = {"|".join(k): v for k, v in all_relation_texts().items()}
    fingerprint = hashlib.sha256(json.dumps(
        {"model": model, "texts": texts, "version": 1}, sort_keys=True
    ).encode()).hexdigest()[:20]
    return Path(root)/f"k7_m1_mechanism_{fingerprint}.pt"


def get_cards(root, model_id, batch_size=2):
    file = card_path(root, model_id)
    names = all_relation_texts()
    if file.exists():
        cache = torch.load(file, map_location="cpu", weights_only=True)
        if set(cache) != {"|".join(k) for k in names}:
            raise ValueError("Incomplete mechanism card cache")
        return {tuple(k.split("|")): F.normalize(v.float().flatten(), dim=0)
                for k, v in cache.items()}
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for first-time Qwen card encoding")
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda").eval()
    keys = sorted(names)
    cache = {}
    with torch.inference_mode():
        for start in tqdm(range(0, len(keys), batch_size),
                          desc="Encode 4 mechanism cards", dynamic_ncols=True):
            batch_keys = keys[start:start+batch_size]
            batch = tokenizer([names[k] for k in batch_keys], padding=True,
                              truncation=True, max_length=1024,
                              return_tensors="pt").to("cuda")
            hidden = model(**batch).last_hidden_state
            vectors = F.normalize(last_token_pool(hidden, batch["attention_mask"]).float(),
                                  dim=-1).cpu()
            for k, v in zip(batch_keys, vectors):
                cache["|".join(k)] = v
    del model
    torch.cuda.empty_cache()
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, file)
    print(f"[k7-m1] saved mechanism cards: {file}", flush=True)
    return {tuple(k.split("|")): F.normalize(v.float().flatten(), dim=0)
            for k, v in cache.items()}


def semantic_inputs(task, embeddings, cards):
    # Train tasks receive only relations to the OTHER train source task.
    # Target receives two permitted source->held-target relation messages.
    if task == HELDOUT_TASK:
        links = [cards[source, task] for source in TRAIN_TASKS]
    elif task in TRAIN_TASKS:
        links = [cards[source, task] for source in TRAIN_TASKS if source != task]
    else:
        raise ValueError("Unexpected task in K7 M1")
    relation = F.normalize(torch.stack(links).mean(0), dim=0)
    return embeddings[task].float(), relation


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("No MI250/ROCm GPU visible in this job")
    if args.seed != 42:
        raise ValueError("K7 M1 fixed pilot is seed=42 only")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Invalid budget")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("K7 M1 output is nonempty. Do not overwrite: " + str(args.output))
    frame, x, ctx, y, names = read_tensor_cache(
        args.tensor_cache, args.source_csv, args.image_root,
        args.feature_dir, args.remoteclip_model)
    if not set((*TRAIN_TASKS, HELDOUT_TASK)).issubset(set(names)):
        raise ValueError("Missing source/validation task in frozen source-only cache")
    print(f"[k7-m1] GPU={torch.cuda.get_device_name(0)} rows={len(frame)}", flush=True)
    embeddings = qwen_task_vectors(
        root=args.text_cache_dir, model_id=args.text_model,
        profile="short", only_cached=True)
    cards = get_cards(args.text_cache_dir, args.text_model)
    cities = sorted(frame.city.astype(str).unique())
    if len(cities) < 3:
        raise ValueError("At least three source cities needed")
    perm = np.random.default_rng(args.seed).permutation(cities)
    n = max(1, min(len(cities)-2, round(len(cities)*args.city_fraction)))
    held_cities = sorted(perm[:n].tolist())
    source_mask = frame.indicator_column.isin(TRAIN_TASKS).to_numpy()
    target_mask = frame.indicator_column.eq(HELDOUT_TASK).to_numpy()
    unseen_city = frame.city.astype(str).isin(held_cities).to_numpy()
    train_idx = np.flatnonzero(source_mask & ~unseen_city)
    seen_val = np.flatnonzero(target_mask & ~unseen_city)
    unseen_val = np.flatnonzero(target_mask & unseen_city)
    if not len(train_idx) or not len(seen_val) or not len(unseen_val):
        raise ValueError("Empty train/validation subset")
    if args.train_limit and args.train_limit < len(train_idx):
        # Stratified sample ONLY training source labels, never validation rows.
        sample = frame.iloc[train_idx].copy().reset_index()
        subset = stratified_sample(sample, args.train_limit, args.seed)
        train_idx = subset["index"].to_numpy(dtype=np.int64)
    train_frame = frame.iloc[train_idx]
    if set(train_frame.indicator_column) != set(TRAIN_TASKS):
        raise ValueError("Training sample lost a source indicator")
    stats, unseen_stats = calibrated_moments(
        train_frame, embeddings, HELDOUT_TASK, args.scale_temperature)
    sem_by_task, rel_by_task = {}, {}
    for task in (*TRAIN_TASKS, HELDOUT_TASK):
        sem_by_task[task], rel_by_task[task] = semantic_inputs(task, embeddings, cards)
    source_ids = torch.as_tensor(train_idx, dtype=torch.long)
    means = torch.tensor([stats[names[i]][0] for i in train_idx], dtype=torch.float32)
    stds = torch.tensor([stats[names[i]][1] for i in train_idx], dtype=torch.float32)
    train_y = (y[source_ids] - means)/stds
    device = torch.device("cuda")
    visual = x[source_ids].to(device)
    context = ctx[source_ids].to(device)
    targets = train_y.to(device)
    semantic = torch.stack([sem_by_task[names[i]] for i in train_idx]).to(device)
    relations = torch.stack([rel_by_task[names[i]] for i in train_idx]).to(device)
    print(f"[k7-m1] source_tasks={TRAIN_TASKS} heldout={HELDOUT_TASK} "
          f"train={len(train_idx)} SCUI={len(seen_val)} UCUI={len(unseen_val)} "
          f"held_cities={len(held_cities)}", flush=True)
    torch.manual_seed(args.seed)
    model = K7HyperRegressor(
        x.shape[1], semantic.shape[1], hidden=args.hidden, rank=args.rank,
        use_relation=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    progress = tqdm(range(args.epochs), desc="K7 M1 single model", dynamic_ncols=True)
    for _ in progress:
        model.train()
        order = torch.randperm(len(train_idx), generator=generator).to(device)
        total, count = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            j = order[start:start+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(visual[j], context[j], semantic[j], relations[j]),
                              targets[j])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            count += 1
        progress.set_postfix(loss=f"{total/max(count,1):.3f}")
    model.eval()
    rows = []
    # Education labels accessed ONLY here, after optimization is complete.
    for split_name, ids in (("SCUI", seen_val), ("UCUI", unseen_val)):
        target_sem = sem_by_task[HELDOUT_TASK].to(device)
        target_rel = rel_by_task[HELDOUT_TASK].to(device)
        parts = []
        with torch.inference_mode():
            for start in range(0, len(ids), args.batch_size):
                j = torch.as_tensor(ids[start:start+args.batch_size], dtype=torch.long)
                p = model(x[j].to(device), ctx[j].to(device),
                          target_sem.expand(len(j), -1),
                          target_rel.expand(len(j), -1))
                parts.append(p.cpu())
        pred = unseen_stats[0] + unseen_stats[1]*torch.cat(parts).numpy()
        truth = y[ids].numpy()
        metric = scores(pred, truth)
        baseline = float(np.mean((unseen_stats[0]-truth)**2))
        result = {
            "partition": split_name, **metric, "source_mean_mse": baseline,
            "relative_mse_to_source_mean": metric["mse_log1p"]/max(baseline, 1e-12),
            **diagnostic(pred, truth),
        }
        rows.append(result)
        print(f"[k7-m1] {split_name} MSE={metric['mse_log1p']:.3f} "
              f"Pearson={metric['pearson']:.3f} "
              f"Ratio={result['relative_mse_to_source_mean']:.3f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save({
        "model": model.cpu().state_dict(), "source_tasks": TRAIN_TASKS,
        "heldout_task": HELDOUT_TASK, "heldout_cities": held_cities,
        "train_stats": stats, "unseen_stats": unseen_stats
    }, args.output/"model.pt")
    (args.output/"config.json").write_text(json.dumps({
        **{k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "heldout_cities": held_cities, "source_tasks": TRAIN_TASKS,
        "heldout_task": HELDOUT_TASK, "train_rows": len(train_idx)
    }, indent=2))
    print(f"[k7-m1] COMPLETE {args.output}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--feature-dir", type=Path, default=Path("runs/m3_features"))
    p.add_argument("--tensor-cache", type=Path,
                   default=Path("runs/k7/e1_e2_v1/_source_tensors.pt"))
    p.add_argument("--remoteclip-model", default="ViT-B-32")
    p.add_argument("--text-cache-dir", type=Path, default=Path("runs/ktrans_text"))
    p.add_argument("--text-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--output", type=Path, default=Path("runs/k7/m1_pop_food_to_education"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-limit", type=int, default=20000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--scale-temperature", type=float, default=0.2)
    p.add_argument("--city-fraction", type=float, default=0.25)
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
