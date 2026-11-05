"""One-command unattended K7 hypernetwork sweep: prepare knowledge, run controls, summarize.

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

from scripts.run_k7_pilot import VARIANTS
from urban_transfer.k7_feature_cache import build_tensor_cache


def prepare_caches(a):
    from urban_transfer.ktrans_text import qwen_task_vectors
    from urban_transfer.z1_relation import relation_embeddings
    print("[k7] checking short Qwen semantic cache...", flush=True)
    vectors = qwen_task_vectors(
        root=a.text_cache_dir, model_id=a.text_model, profile="short",
        only_cached=False, batch_size=a.text_batch_size,
        device=a.embedding_device)
    print(f"[k7] short embeddings: {len(vectors)} indicators", flush=True)
    if any(v != "k7_e1" for v in a.variants):
        pairs = relation_embeddings(
            a.relation_cache_dir, a.text_model, only_cached=False,
            device=a.embedding_device, batch_size=a.text_batch_size)
        print(f"[k7] relation embeddings: {len(pairs)} directed pairs", flush=True)
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
    lines = ["# K7 source-only DEV report", "",
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
    import torch
    if not a.cpu and not torch.cuda.is_available():
        raise RuntimeError(
            "No ROCm/CUDA GPU visible in this job; submit inside a GPU allocation "
            "or pass --cpu explicitly. Refusing silent CPU training.")
    print(f"[k7-night] training_device={'CPU (explicit)' if a.cpu else torch.cuda.get_device_name(0)} "
          f"ROCm={torch.version.hip} CUDA={torch.version.cuda}", flush=True)
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
    # Build once for ALL experiments, avoiding thousands of redundant SQLite
    # feature lookups. Reuse same cache on resume without modifying fold configs.
    shared = a.output/"_source_tensors.pt"
    build_tensor_cache(a.source_csv, a.image_root, a.feature_dir,
                       a.remoteclip_model, shared, a.image_column)
    print(f"[k7-night] shared frozen tensors ready: {shared}", flush=True)
    status = []
    failed = False
    progress = tqdm(jobs, desc="K7 hypernetwork sweep", dynamic_ncols=True)
    for variant, seed, directory in progress:
        directory.mkdir(parents=True, exist_ok=True)
        log = directory/"console.log"
        cmd = [
            sys.executable, "-u", "scripts/run_k7_pilot.py",
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
            f.write("\n==== K7 RUN: "+json.dumps(cmd)+" ====\n")
            f.flush()
            env = os.environ.copy()
            env["K7_SHARED_TENSOR_CACHE"] = str(shared.resolve())
            # Stream internal Epoch progress rather than leaving the outer
            # progress bar apparently stuck at 0% for the entire first run.
            with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True,
                                  bufsize=1, env=env,
                                  cwd=Path(__file__).resolve().parents[1]) as proc:
                for line in proc.stdout:
                    f.write(line)
                    f.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                returncode = proc.wait()
        if returncode:
            state = f"FAILED exit={returncode}"
            failed = True
            print(f"[k7-night] {state}: {variant} seed={seed}; {log}", flush=True)
            # Continue independent experiments so one failure does not erase the night.
        else:
            print(f"[k7-night] OK {variant} seed={seed}", flush=True)
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
    p.add_argument("--output", type=Path, default=Path("runs/k7/e1_e2_v1"))
    p.add_argument("--variants", nargs="+", choices=VARIANTS,
                   default=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--rank", type=int, default=8)
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
