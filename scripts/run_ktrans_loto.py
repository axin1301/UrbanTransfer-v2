"""Six-source-task LOTO: one expert checkpoint per fold, reused across routes.

Run: python scripts/run_ktrans_loto.py --mode all
Summarize: python scripts/run_ktrans_loto.py --mode summarize
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    'population_count', 'nighttime_light_intensity',
    'education_accessibility', 'food_accessibility',
    'digital_modernization_tmi', 'digital_total_cell_intensity',
)
PROTOCOLS = ('meta_SCUI', 'meta_UCSI', 'meta_UCUI')


def run_command(argv):
    print('[LOTO]', ' '.join(map(str, argv)), flush=True)
    subprocess.run(argv, cwd=ROOT, check=True)


def paths(root, task, route):
    return root / task / route / 'diagnostics.csv'


def summarize(root, seeds, output):
    rows = []
    for task in TASKS:
        for mode in ('semantic', 'uniform'):
            path = paths(root, task, mode)
            if not path.is_file():
                raise FileNotFoundError(f'Missing {path}; run --mode evaluate first')
            with path.open(newline='') as file:
                records = list(csv.DictReader(file))
            for item in records:
                rows.append({
                    'task': task, 'route': mode, 'protocol': item['protocol'],
                    'seed': '', 'mse': float(item['mse']),
                    'r2': float(item['r2']) if item['r2'] else None,
                    'bias': float(item['bias']),
                    'pearson': float(item['pearson']) if item['pearson'] else None,
                })
        for seed in seeds:
            path = paths(root, task, f'random_{seed}')
            if not path.is_file():
                raise FileNotFoundError(f'Missing {path}; run --mode evaluate first')
            with path.open(newline='') as file:
                for item in csv.DictReader(file):
                    rows.append({
                        'task': task, 'route': 'random', 'protocol': item['protocol'],
                        'seed': seed, 'mse': float(item['mse']),
                        'r2': float(item['r2']) if item['r2'] else None,
                        'bias': float(item['bias']),
                        'pearson': float(item['pearson']) if item['pearson'] else None,
                    })
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'all_runs.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    # Aggregate across held-out tasks. For Random, average random seeds within
    # a fold FIRST: multiple draws must not give random more fold weight.
    import statistics
    agg = []
    for mode in ('semantic', 'uniform', 'random'):
        for protocol in PROTOCOLS:
            fold_rows = []
            for task in TASKS:
                items = [r for r in rows if r['route'] == mode and
                         r['protocol'] == protocol and r['task'] == task]
                fold_rows.append({
                    key: statistics.mean(r[key] for r in items if r[key] is not None)
                    for key in ('mse', 'r2', 'bias', 'pearson')
                    if any(r[key] is not None for r in items)
                })
            agg.append({
                'route': mode, 'protocol': protocol,
                'n_folds': len(fold_rows),
                'mse_mean': statistics.mean(x['mse'] for x in fold_rows),
                'mse_std': statistics.stdev(x['mse'] for x in fold_rows),
                'r2_mean': statistics.mean(x['r2'] for x in fold_rows if 'r2' in x),
                'pearson_mean': statistics.mean(x['pearson'] for x in fold_rows if 'pearson' in x),
                'abs_bias_mean': statistics.mean(abs(x['bias']) for x in fold_rows),
            })
    with (output / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=agg[0].keys())
        writer.writeheader()
        writer.writerows(agg)
    print(f"{'Route':<10} {'Protocol':<11} {'MSE':>9} {'±SD':>9} {'R2':>9} {'Pearson':>9} {'|Bias|':>9}")
    for r in agg:
        print(f"{r['route']:<10} {r['protocol']:<11} "
              f"{r['mse_mean']:>9.3f} {r['mse_std']:>9.3f} "
              f"{r['r2_mean']:>9.3f} {r['pearson_mean']:>9.3f} "
              f"{r['abs_bias_mean']:>9.3f}")
    print(f'[LOTO] saved {output / "summary.csv"} and {output / "all_runs.csv"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all', 'train', 'evaluate', 'summarize'], default='all')
    p.add_argument('--task-encoder',choices=['remoteclip','qwen3'],default='remoteclip')
    p.add_argument('--task-profile',choices=['short','structured','llm'],default='short')
    p.add_argument('--text-model-id',default='Qwen/Qwen3-Embedding-0.6B')
    p.add_argument('--llm-profile-file')
    p.add_argument('--normalized-experts',action='store_true',help='K3-A: standardized source experts; use fresh runs-root')
    p.add_argument('--limit', type=int, default=5000)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--eval-batch-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--random-seeds', type=int, nargs='+', default=[41, 42, 43, 44, 45])
    p.add_argument('--runs-root', type=Path, default=Path('runs/ktrans_loto'))
    p.add_argument('--feature-dir', default='runs/m3_features')
    p.add_argument('--text-cache-dir', default='runs/ktrans_text_cache')
    args = p.parse_args()
    if len(set(args.random_seeds)) != len(args.random_seeds):
        p.error('--random-seeds must be distinct')
    runs_root = (ROOT / args.runs_root).resolve()
    if args.mode in ('all', 'train'):
        for task in TASKS:
            ckpt = runs_root / task / 'best.pt'
            if ckpt.exists():
                print(f'[LOTO] reuse existing {ckpt}', flush=True)
                continue
            run_command([
                sys.executable, '-u', 'run_ktrans.py', 'train',
                *(['--normalized-experts'] if args.normalized_experts else []),
                '--expert-bank', '--task-encoder', args.task_encoder,
                '--task-profile', args.task_profile, '--text-model-id', args.text_model_id,
                *(['--llm-profile-file', args.llm_profile_file] if args.llm_profile_file else []),
                '--heldout-task', task,
                '--limit', str(args.limit), '--epochs', str(args.epochs),
                '--patience', '0', '--seed', str(args.seed),
                '--batch-size', str(args.batch_size),
                '--eval-batch-size', str(args.eval_batch_size),
                '--feature-dir', args.feature_dir,
                '--text-cache-dir', args.text_cache_dir,
                '--output', str(runs_root / task),
            ])
    if args.mode in ('all', 'evaluate'):
        for task in TASKS:
            ckpt = runs_root / task / 'best.pt'
            if not ckpt.is_file():
                raise FileNotFoundError(f'Expected trained checkpoint: {ckpt}')
            for route, seed in [('semantic', args.seed), ('uniform', args.seed)] + [
                ('random', s) for s in args.random_seeds
            ]:
                tag = f'random_{seed}' if route == 'random' else route
                dest = runs_root / 'routing' / task / tag
                if (dest / 'diagnostics.csv').is_file():
                    print(f'[LOTO] reuse existing {dest}', flush=True)
                    continue
                run_command([
                    sys.executable, '-u', 'scripts/analyze_ktrans_predictions.py',
                    '--checkpoint', str(ckpt),
                    *(['--llm-profile-file', args.llm_profile_file] if args.llm_profile_file else []),
                    '--routing-mode', route, '--routing-seed', str(seed),
                    '--feature-dir', args.feature_dir,
                    '--text-cache-dir', args.text_cache_dir,
                    '--batch-size', str(args.eval_batch_size),
                    '--output', str(dest),
                ])
    if args.mode in ('all', 'summarize'):
        summarize(runs_root / 'routing', args.random_seeds, runs_root / 'summary')


if __name__ == '__main__':
    main()
