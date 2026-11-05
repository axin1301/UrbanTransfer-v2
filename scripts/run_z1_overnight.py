"""One-command unattended Z1 sweep: prepare knowledge, run controls, summarize.

Run as an HPC batch job. Outputs, logs and per-fold checkpoints persist,
and rerunning with identical parameters resumes incomplete experiments.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm.auto import tqdm

from scripts.run_z1_relation import VARIANTS


def prepare_caches(a):
    from urban_transfer.ktrans_text import qwen_task_vectors
    from urban_transfer.z1_relation import relation_embeddings
    print("[z1-night] checking frozen Qwen text caches...", flush=True)
    for profile in ("short", "structured"):
        vectors = qwen_task_vectors(root=a.text_cache_dir, model_id=a.text_model,
                                   profile=profile, only_cached=False,
                                   batch_size=a.text_batch_size,
                                   device=a.embedding_device)
        print(f"[z1-night] {profile} embeddings: {len(vectors)} indicators", flush=True)
    if any(v.startswith("relation_") for v in a.variants):
        relations = relation_embeddings(
            a.relation_cache_dir, a.text_model, only_cached=False,
            device=a.embedding_device, batch_size=a.text_batch_size)
        print(f"[z1-night] relation embeddings: {len(relations)} pairs", flush=True)
    # Release embedding-model resources before small-model train subprocesses.
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def collect(a, jobs):
    rows = []
    for variant, seed, directory in jobs:
        file = directory/"macro_metrics.csv"
        if not file.exists():
            continue
        with file.open() as f:
            for row in csv.DictReader(f):
                rows.append({**row, "run": str(directory)})
    if not rows:
        return []
    path = a.output/"sweep_macro.csv"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return rows


def summarize(a, jobs, status):
    rows = collect(a, jobs)
    metrics = {}
    for row in rows:
        if row["partition"] != "UCUI":
            continue
        name = row["variant"]
        metrics.setdefault(name, []).append(row)
    lines = ["# Z1 unattended source-only DEV report", "",
             "All results use source pseudo-indicator folds, not final target tests.",
             "Oracle label-fitted diagnostics are never zero-shot performance.",
             "",
             "| Variant | Runs | UCUI MSE | UCUI Pearson | UCUI ratio |",
             "|---|---:|---:|---:|---:|"]
    for variant in a.variants:
        group = metrics.get(variant, [])
        if not group:
            lines.append(f"| {variant} | 0 | N/A | N/A | N/A |")
            continue
        means = [np.mean([float(r[key]) for r in group]) for key in
                 ("mean_mse_log1p", "mean_pearson", "mean_relative_mse")]
        lines.append(f"| {variant} | {len(group)} | " +
                     " | ".join(f"{v:.3f}" for v in means) + " |")
    lines += ["", "## Job status", ""]
    for item in status:
        lines.append(f"- {item['variant']} seed={item['seed']}: "
                     f"{item['state']} (log: {item['log']})")
    (a.output/"REPORT.md").write_text("\n".join(lines)+"\n")
    (a.output/"job_status.json").write_text(json.dumps(status, indent=2))
    print("\n".join(lines[:len(a.variants)+8]), flush=True)


def run(a):
    if not a.source_csv.is_file():
        raise FileNotFoundError("Source CSV not found: " + str(a.source_csv))
    if not a.image_root.is_dir():
        raise FileNotFoundError("Image root not found: " + str(a.image_root))
    if not (a.feature_dir/"remoteclip_features.sqlite3").is_file():
        raise FileNotFoundError("RemoteCLIP SQLite cache not found under: " + str(a.feature_dir))
    a.output.mkdir(parents=True, exist_ok=True)
    jobs = [(variant, seed, a.output/f"{variant}_seed{seed}")
            for seed in a.seeds for variant in a.variants]
    if not a.skip_prepare:
        prepare_caches(a)
    status = []
    failed = False
    progress = tqdm(jobs, desc="Z1 unattended sweep", dynamic_ncols=True)
    for variant, seed, directory in progress:
        directory.mkdir(parents=True, exist_ok=True)
        log = directory/"console.log"
        cmd = [
            sys.executable, "-u", "scripts/run_z1_relation.py",
            "--variant", variant, "--seed", str(seed),
            "--source-csv", str(a.source_csv),
            "--image-root", str(a.image_root),
            "--feature-dir", str(a.feature_dir),
            "--remoteclip-model", a.remoteclip_model,
            "--text-model", a.text_model,
            "--text-cache-dir", str(a.text_cache_dir),
            "--relation-cache-dir", str(a.relation_cache_dir),
            "--output", str(directory),
            "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
            "--hidden", str(a.hidden), "--rank", str(a.rank),
            "--lr", str(a.lr), "--city-fraction", str(a.city_fraction)]
        if a.image_column:
            cmd += ["--image-column", a.image_column]
        if a.cpu:
            cmd.append("--cpu")
        state = "OK"
        with log.open("a") as f:
            f.write("\n==== Z1 RUN: "+json.dumps(cmd)+" ====\n")
            f.flush()
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                    cwd=Path(__file__).resolve().parents[1],
                                    check=False)
        if result.returncode:
            state = f"FAILED exit={result.returncode}"
            failed = True
            print(f"[z1-night] {state}: {variant} seed={seed}; {log}", flush=True)
            # Continue independent experiments so one failure does not erase the night.
        else:
            print(f"[z1-night] OK {variant} seed={seed}", flush=True)
        status.append({"variant": variant, "seed": seed, "state": state,
                       "log": str(log)})
        summarize(a, jobs, status)
        progress.set_postfix(done=len(status), failed=sum(
            item["state"].startswith("FAILED") for item in status))
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--image-column")
    p.add_argument("--feature-dir", type=Path, required=True)
    p.add_argument("--remoteclip-model", default="ViT-B-32")
    p.add_argument("--text-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--text-cache-dir", type=Path, default=Path("runs/ktrans_text"))
    p.add_argument("--relation-cache-dir", type=Path, default=Path("runs/ktrans_text"))
    p.add_argument("--output", type=Path, default=Path("runs/z1/overnight_v1"))
    p.add_argument("--variants", nargs="+", choices=VARIANTS,
                   default=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--city-fraction", type=float, default=0.25)
    p.add_argument("--text-batch-size", type=int, default=2)
    p.add_argument("--embedding-device", choices=("cpu", "cuda"))
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    if len(a.variants) != len(set(a.variants)) or len(a.seeds) != len(set(a.seeds)):
        p.error("Duplicate variants/seeds not allowed")
    if a.epochs < 1 or a.batch_size < 1 or a.text_batch_size < 1:
        p.error("Invalid budget")
    try:
        return run(a)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
