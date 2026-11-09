"""M6.5 source-city ablation: identical-capacity, NO neighborhood information.

Single matched-control fit on the exact existing M6.5 dev city partition,
reusing its separately trained, leak-free dev_visual_anchor.pt.
Use the SAME NeighborhoodCityIndicatorResidual class, 20 epochs,
batch256, objective and Qwen token cache as the real-neighborhood dev
run. Only input changes: all neighbor features are replaced by zeros
in TRAIN and VALIDATION. All trainable layers remain enabled, including
the neighborhood encoder, so parameter count is identical.

This contrasts a retrained no-neighborhood model vs existing M6.5
source-held-city metrics. Unlike inference-time knockouts, it does not
subject a trained model to out-of-distribution ablation inputs.

NOTE: original dev model was initialized after its visual-anchor training
consumed RNG. Therefore the control uses a reproducible seed42 model
initialization, but it is not bitwise matched to original M6.5 initial
weights. Report as single-seed same-architecture evidence, not causal proof.
No Education label or official test file is read.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts.run_k7_m62_full import CachedTokens,validate_full
from scripts.run_k7_m63_full import PersistentReadOnlyHiddenStore
from scripts.run_k7_m65_full import (
    city_dev_split,fit_source_moments,train_neighbor_student,
    evaluate_source_or_target)
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS
from urban_transfer.k7_m65_neighborhood_student import (
    NeighborhoodCityIndicatorResidual)


def assert_same_anchor(source,model):
    base_state=model.base.state_dict()
    if set(source)!=set(base_state):
        raise ValueError("Dev visual anchor architecture mismatch")
    for key,value in source.items():
        if not torch.equal(value.detach().cpu(),base_state[key].detach().cpu()):
            raise ValueError("Dev visual anchor must be frozen: "+key)


def source_city_comparison(reference_rows,control_rows):
    by_partition={row["partition"]:row for row in reference_rows}
    output=[]
    for control in control_rows:
        key=control["partition"]
        if key not in by_partition:
            raise ValueError("Original M6.5 dev result missing "+key)
        real=by_partition[key]
        if int(real["n_rows"])!=int(control["n_rows"]):
            raise ValueError("Dev source held-city sample mismatch "+key)
        for metric in ("mse_log1p","mae_log1p","pearson",
                       "mean_bias_log1p","population_weight"):
            a=float(real[metric])
            b=float(control[metric])
            output.append({
                "partition":key,"metric":metric,
                "real_neighborhood":a,"zero_neighborhood":b,
                "real_minus_zero":a-b,
            })
    return output


def display_rows_3dp(rows):
    """Round only serialized floating metrics; retain full precision in analysis."""
    return [
        {key:(f"{value:.3f}" if isinstance(value,(float,np.floating))
              else value)
         for key,value in row.items()}
        for row in rows
    ]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-dir",type=Path,
                   default=Path("runs/k7/m62_full_seed42"))
    p.add_argument("--m65-dev-dir",type=Path,
                   default=Path("runs/k7/m65_neighborhood_dev_seed42"))
    p.add_argument("--output",type=Path,
                   default=Path("runs/k7/m65_ablation_zero_neighborhood_dev_seed42"))
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--epochs",type=int,default=20)
    args=p.parse_args()
    if args.batch_size!=256 or args.epochs!=20:
        p.error("Matched M6.5 control requires 20 epochs and batch256")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU not available")
    if args.output.resolve()==args.m65_dev_dir.resolve():
        raise ValueError("Use a NEW ablation output directory; never overwrite M6.5")
    checkpoint_path=args.output/"checkpoint_last.pt"
    if (args.output/"metrics.csv").exists():
        raise FileExistsError("Ablation already complete; refusing retrain")
    if args.output.exists() and not checkpoint_path.is_file():
        raise FileExistsError(
            "Existing ablation output without resumable checkpoint")

    manifest_path=(args.full_dir/"full_manifest.json").resolve()
    manifest=json.loads(manifest_path.read_text())
    frame,x,ctx,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    splits=validate_full(frame,manifest)
    train_ids,held_ids,held_cities=city_dev_split(frame,splits["train"])
    source_stats=fit_source_moments(y,names,train_ids)
    source_sha=hashlib.sha256(
        np.asarray(train_ids,dtype=np.int64).tobytes()).hexdigest()

    # Original M6.5 dev results are immutable reference, NOT calibration
    # or model selection data; held source labels are used only to report.
    dev_config=json.loads((args.m65_dev_dir/"config.json").read_text())
    if (dev_config["stage"]!="dev"
        or dev_config["source_train_sha256"]!=source_sha
        or dev_config["held_source_cities"]!=held_cities
        or Path(dev_config["manifest"]).resolve()!=manifest_path
        or int(dev_config["batch_size"])!=256
        or int(dev_config["epochs"])!=20):
        raise ValueError("Original M6.5 dev protocol/split mismatch")

    anchor_saved=torch.load(
        args.m65_dev_dir/"dev_visual_anchor.pt",
        map_location="cpu",weights_only=True)
    original_model=torch.load(
        args.m65_dev_dir/"m65.pt",map_location="cpu",weights_only=True)
    if (original_model["config"]!={
        k:dev_config[k] for k in original_model["config"]
    } or original_model["source_stats"]!=source_stats):
        raise ValueError("Original M6.5 dev checkpoint metadata mismatch")
    original_state=original_model["model"]
    anchor_state=anchor_saved["model"]
    for key,value in anchor_state.items():
        if not torch.equal(original_state["base."+key],value):
            raise ValueError("Original M6.5 dev checkpoint anchor differs")
    anchor_signature=anchor_saved["signature"]
    if (anchor_signature["train_hash"]!=source_sha
        or Path(anchor_signature["manifest"]).resolve()!=manifest_path
        or anchor_signature["source_stats"]!=source_stats
        or int(anchor_signature["batch_size"])!=256
        or int(anchor_signature["epochs"])!=20):
        raise ValueError("Dev anchor has seen wrong source cities")

    with (args.m65_dev_dir/"metrics.csv").open(newline="") as handle:
        reference_rows=list(csv.DictReader(handle))
    expected={"dev_held_city_"+task for task in TRAIN_TASKS}
    if set(row["partition"] for row in reference_rows)!=expected:
        raise ValueError("Original Dev results must contain only Pop/Food")

    ds=ImageUrbanDataset(
        manifest["source_csv"],manifest["image_root"],partition="eval")
    if len(ds.df)!=len(frame) or names!=ds.df.indicator_column.astype(str).tolist():
        raise ValueError("Image/token row ordering mismatch")

    store=PersistentReadOnlyHiddenStore(args.full_dir/"hidden.sqlite3")
    try:
        tokens=CachedTokens(frame,ds,manifest,store,maxsize=512)
        first=tokens[int(train_ids[0]),names[int(train_ids[0])]]
        torch.manual_seed(42)
        if args.device=="cuda":
            torch.cuda.manual_seed_all(42)
        model=NeighborhoodCityIndicatorResidual(
            x.shape[-1],first.shape[-1],hidden=128,heads=4,
            group_counts=tuple(manifest["group_counts"]),
            max_residual=.5).to(args.device)
        model.load_visual_baseline(anchor_state)
        for parameter in model.base.parameters():
            parameter.requires_grad_(False)
        assert_same_anchor(anchor_state,model)
        neighborhood=torch.zeros(
            len(frame),model.neighborhood_dim,dtype=torch.float32)
        signature={
            "stage":"dev_zero_neighborhood",
            "manifest":str(manifest_path),
            "visual_anchor":str(
                (args.m65_dev_dir/"dev_visual_anchor.pt").resolve()),
            "anchor_train_hash":source_sha,
            "source_train_sha256":source_sha,
            "held_source_cities":held_cities,
            "epochs":20,"batch_size":256,"seed":42,
            "mode":"same_architecture_zero_neighbor_in_train_and_eval_v1",
        }
        args.output.mkdir(parents=True,exist_ok=True)
        print(f"[k7-m65-ablate] START source_train={len(train_ids)} "
              f"held_source={len(held_ids)} cities={held_cities} "
              f"frozen_anchor=reused no_neighborhood=True",flush=True)
        train_neighbor_student(
            model,x,ctx,neighborhood,y,names,train_ids,source_stats,
            tokens,args,torch.device(args.device),checkpoint_path,signature)
        rows=[]
        for task in TRAIN_TASKS:
            ids=np.asarray([int(i) for i in held_ids
                            if names[int(i)]==task],dtype=np.int64)
            row=evaluate_source_or_target(
                model,"dev_held_city_"+task,ids,x,ctx,neighborhood,
                y,names,source_stats,(0.,1.),tokens,args.batch_size,
                torch.device(args.device))
            row["model"]="m65_zero_neighborhood"
            rows.append(row)
            print(f"[k7-m65-ablate] {task} n={len(ids)} "
                  f"MSE={row['mse_log1p']:.3f} "
                  f"Pearson={row['pearson']:.3f} "
                  f"bias={row['mean_bias_log1p']:+.3f}",flush=True)
        compare=source_city_comparison(reference_rows,rows)
        for row in compare:
            if row["metric"] in ("mse_log1p","pearson"):
                print(f"[k7-m65-ablate] comparison {row['partition']} "
                      f"{row['metric']} real={row['real_neighborhood']:.3f} "
                      f"zero={row['zero_neighborhood']:.3f} "
                      f"real_minus_zero={row['real_minus_zero']:+.3f}",
                      flush=True)
        with (args.output/"metrics.csv").open("w",newline="") as output:
            writer=csv.DictWriter(output,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(display_rows_3dp(rows))
        with (args.output/"comparison.csv").open("w",newline="") as output:
            writer=csv.DictWriter(output,fieldnames=list(compare[0]))
            writer.writeheader()
            writer.writerows(display_rows_3dp(compare))
        (args.output/"config.json").write_text(
            json.dumps({**signature,"reference_dev":str(
                args.m65_dev_dir.resolve()),
                "caveat":"Original dev initialization RNG followed full "
                "visual anchor training; initialization is not bitwise matched.",
                "target_labels_used":False},indent=2))
        print(f"[k7-m65-ablate] COMPLETE {args.output}",flush=True)
    finally:
        store.close()


if __name__=="__main__":
    main()
