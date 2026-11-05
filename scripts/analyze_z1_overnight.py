"""Post-hoc UCUI diagnostics from existing Z1 folds: no training, no images.

The oracle columns access held-out DEV labels: NOT zero-shot performance.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

METRICS = ("mse_log1p", "pearson", "relative_mse_to_source_mean",
           "prediction_bias_oracle", "pred_std_to_true_std_oracle",
           "affine_fitted_mse_ORACLE_NOT_ZERO_SHOT")
PAIRS = (
    ("structured_b1", "short_b1"),
    ("structured_b2", "structured_b1"),
    ("structured_b3", "structured_b1"),
    ("relation_b1", "structured_b1"),
    ("relation_b2", "structured_b2"),
    ("relation_b3", "structured_b3"),
    ("relation_b1", "relation_shuffle_b1"),
    ("relation_b2", "relation_shuffle_b2"),
    ("relation_b3", "relation_shuffle_b3"),
)


def load(root):
    result = {}
    for path in sorted(root.glob("*_seed*/fold_metrics.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["partition"] != "UCUI":
                    continue
                key = (row["variant"], int(row["seed"]), row["heldout_task"])
                if key in result:
                    raise ValueError("Duplicate fold " + repr(key))
                result[key] = {m: float(row[m]) for m in METRICS}
    if not result:
        raise FileNotFoundError("No completed UCUI fold_metrics.csv in " + str(root))
    return result


def avg(numbers):
    finite = [v for v in numbers if math.isfinite(v)]
    return mean(finite) if finite else float("nan")


def fmt(value):
    return f"{value:.3f}" if math.isfinite(value) else "N/A"


def summary(data):
    lines = [
        "| Variant | Folds | MSE | Pearson | Ratio | Signed bias | Std ratio | Affine oracle MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in sorted({key[0] for key in data}):
        records = [row for (v, _, _), row in data.items() if v == variant]
        values = [avg([r[m] for r in records]) for m in METRICS]
        lines.append(f"| {variant} | {len(records)} | " +
                     " | ".join(map(fmt, values)) + " |")
    return lines


def paired(data):
    lines = [
        "| A versus B | Matched folds | ΔMSE | ΔPearson | ΔRatio | Δabs(bias) | Δoracle MSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b in PAIRS:
        shared = sorted((seed, task) for (v, seed, task) in data
                        if v == a and (b, seed, task) in data)
        diffs = defaultdict(list)
        for seed, task in shared:
            lhs, rhs = data[a, seed, task], data[b, seed, task]
            for m in METRICS:
                diffs[m].append(lhs[m] - rhs[m])
            diffs["absolute_bias"].append(
                abs(lhs["prediction_bias_oracle"]) -
                abs(rhs["prediction_bias_oracle"]))
        keys = ("mse_log1p", "pearson", "relative_mse_to_source_mean",
                "absolute_bias", "affine_fitted_mse_ORACLE_NOT_ZERO_SHOT")
        lines.append(f"| {a} vs {b} | {len(shared)} | " +
                     " | ".join(fmt(avg(diffs[k])) for k in keys) + " |")
    return lines


def per_task(data):
    grouped = defaultdict(list)
    for (variant, seed, task), record in data.items():
        grouped[(task, variant)].append(record)
    lines = [
        "| Task | Variant | Seeds | MSE | Pearson | Ratio | Bias | Std ratio | Oracle MSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (task, variant), records in sorted(grouped.items()):
        values = [avg([r[m] for r in records]) for m in METRICS]
        lines.append(f"| {task} | {variant} | {len(records)} | " +
                     " | ".join(map(fmt, values)) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/z1/overnight_relation_v1"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = load(args.input)
    output = args.output or args.input/"DIAGNOSTICS.md"
    lines = [
        "# Z1 UCUI diagnostics — existing runs only", "",
        "**No retraining. Oracle columns use held-out DEV labels and must never "
        "be reported as deployable zero-shot performance.**", "",
        "## Per-condition means (each fold weighted equally)", "",
        *summary(data), "",
        "## Paired contrasts (same seed and held-out task)", "",
        "Positive ΔMSE and negative ΔPearson indicate worse A. "
        "Differences are descriptive, not statistical significance tests.", "",
        *paired(data), "",
        "## Per-task results", "", *per_task(data), "",
        "## Important limitations", "",
        "- Structured semantics also change calibrated_moments() and hence "
        "the source-only scale prior; the comparison does not isolate regression.",
        "- Oracle affine MSE fits to evaluation labels and is strictly diagnostic.",
        "- Mean signed bias can cancel; check individual tasks and std ratios.",
        "- Two seeds do not establish statistical significance.",
        "- These are source pseudo-target DEV folds, not final target tests.", "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"[z1-diagnostic] {len(data)} UCUI folds -> {output}", flush=True)
    print("\n".join(paired(data)))


if __name__ == "__main__":
    main()
