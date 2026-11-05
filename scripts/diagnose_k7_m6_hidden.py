"""Read-only M6 hidden-state and router diagnostic; NO training or MLLM calls.

Uses exact M6 training subset, checkpoint and prompt-keyed SQLite entries.
Never reads target labels or official evaluation datasets.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.k7_mllm_router import MLLMHiddenRouter,mllm_teacher_distribution
from urban_transfer.k7_mllm_hidden import HiddenStore,TASKS
from scripts.run_k7_m6_student import load_precomputed


def summary(values):
    a=np.asarray(values,dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite diagnostic scores")
    return {"mean":float(a.mean()),"std":float(a.std()),
            "min":float(a.min()),"max":float(a.max())}


def summarize_hidden(source,target,temperature):
    """Raw cosine, teacher distribution, and image-dependent variability."""
    if source.ndim!=4 or target.ndim!=3:
        raise ValueError("Expected [N,2,L,D] source and [N,L,D] target")
    source_mean=F.normalize(source.float().mean(-2),dim=-1)
    target_mean=F.normalize(target.float().mean(-2),dim=-1)
    cos=(source_mean*target_mean[:,None]).sum(-1).numpy()
    teacher=mllm_teacher_distribution(source,target,temperature=temperature).numpy()
    delta=cos[:,0]-cos[:,1]
    return cos,teacher,{
        "pop_education_cos":summary(cos[:,0]),
        "food_education_cos":summary(cos[:,1]),
        "pop_minus_food_cos":summary(delta),
        "teacher_pop":summary(teacher[:,0]),
        "teacher_food":summary(teacher[:,1]),
        "teacher_pop_percentiles":[float(x) for x in np.quantile(teacher[:,0],[0,.25,.5,.75,1])],
        "fraction_teacher_pop_gt_0_55":float((teacher[:,0]>.55).mean()),
        "fraction_teacher_pop_lt_0_45":float((teacher[:,0]<.45).mean()),
    }


def load_model(checkpoint,config,visual_dim,hidden_dim):
    model=MLLMHiddenRouter(visual_dim,hidden_dim,
        hidden=int(config["hidden"]),heads=int(config["heads"]))
    model.load_state_dict(checkpoint["model"],strict=True)
    return model.eval()


@torch.inference_mode()
def student_diagnostic(model,visual,context,target,source,batch_size):
    weights=[]
    predictions=[]
    swapped=[]
    for start in range(0,len(visual),batch_size):
        x=visual[start:start+batch_size]
        ctx=context[start:start+batch_size]
        tokens=target[start:start+batch_size]
        p,w=model(x,ctx,tokens,return_weights=True)
        # Image/text mismatch intervention only on SOURCE TRAIN rows.
        # Permuting MLLM tokens leaves image/geo inputs exactly unchanged.
        rotated=target[(torch.arange(start,start+len(x))+1)%len(visual)]
        p_swap,w_swap=model(x,ctx,rotated,return_weights=True)
        predictions.append(p)
        weights.append(w)
        swapped.append((p_swap,w_swap))
    w=torch.cat(weights).numpy()
    pred=torch.cat(predictions).numpy()
    swap_pred=torch.cat([p for p,_ in swapped]).numpy()
    swap_w=torch.cat([w for _,w in swapped]).numpy()
    return w,pred,swap_w,swap_pred


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,default=Path("runs/k7/m6_mllm_hidden_seed42"))
    p.add_argument("--output",type=Path,default=None)
    p.add_argument("--batch-size",type=int,default=64)
    a=p.parse_args()
    if a.batch_size<1:p.error("positive batch size required")
    run=a.run_dir
    manifest_file=run/"manifest.json"
    config_file=run/"config.json"
    checkpoint_file=run/"model.pt"
    for path in (manifest_file,config_file,checkpoint_file,run/"hidden.sqlite3"):
        if not path.is_file():raise FileNotFoundError(path)
    manifest=json.loads(manifest_file.read_text())
    config=json.loads(config_file.read_text())
    if int(config["seed"])!=42 or tuple(config["source_tasks"])!=TRAIN_TASKS or config["heldout_task"]!=HELDOUT_TASK:
        raise ValueError("Incompatible M6 checkpoint/protocol")
    indices=np.asarray(manifest["train"],dtype=np.int64)
    frame,visual,context,_,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    if (not set(names[i] for i in indices).issubset(TRAIN_TASKS) or
        frame.city.astype(str).iloc[indices].isin(manifest["held_cities"]).any()):
        raise ValueError("Diagnostic restricted to source-only training rows")
    ds=ImageUrbanDataset(manifest["source_csv"],manifest["image_root"],partition="eval")
    if len(ds.df)!=len(frame) or ds.df.indicator_column.astype(str).tolist()!=names:
        raise ValueError("Image/cache row mismatch")
    cache=load_precomputed(
        frame,indices,TASKS,ds,HiddenStore(run/"hidden.sqlite3"),manifest)
    sources=torch.stack([torch.stack([
        cache[int(i),source] for source in TRAIN_TASKS]) for i in indices])
    target=torch.stack([cache[int(i),HELDOUT_TASK] for i in indices])
    cos,teacher,teacher_stats=summarize_hidden(
        sources,target,float(config["teacher_temperature"]))
    checkpoint=torch.load(checkpoint_file,map_location="cpu",weights_only=True)
    model=load_model(checkpoint,config,visual.shape[1],target.shape[-1])
    w,pred,w_swap,pred_swap=student_diagnostic(
        model,visual[indices],context[indices],target,sources,a.batch_size)
    report={
        "protocol":"READ ONLY; source train rows only; no Education labels or official target tests",
        "n":len(indices),"hidden_shape":list(target.shape[1:]),
        "teacher":teacher_stats,
        "student_pop":summary(w[:,0]),
        "student_food":summary(w[:,1]),
        "teacher_student_pop_abs_diff":summary(np.abs(w[:,0]-teacher[:,0])),
        "teacher_student_pop_correlation":(
            float(np.corrcoef(w[:,0],teacher[:,0])[0,1])
            if w[:,0].std()>1e-9 and teacher[:,0].std()>1e-9 else None),
        "prediction_abs_change_on_hidden_mismatch":summary(np.abs(pred-pred_swap)),
        "pop_weight_abs_change_on_hidden_mismatch":summary(np.abs(w[:,0]-w_swap[:,0])),
    }
    output=a.output or run/"hidden_router_diagnostic.json"
    if output.exists():raise FileExistsError("Refusing overwrite: "+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    print(f"[k7-m6-diag] n={len(indices)} tokens={target.shape[1]} dim={target.shape[2]}")
    print(f"{'Quantity':34} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    fields={
        "cos(Pop,Education)":report["teacher"]["pop_education_cos"],
        "cos(Food,Education)":report["teacher"]["food_education_cos"],
        "cos difference Pop-Food":report["teacher"]["pop_minus_food_cos"],
        "teacher Population weight":report["teacher"]["teacher_pop"],
        "student Population weight":report["student_pop"],
        "abs teacher-student Pop":report["teacher_student_pop_abs_diff"],
        "|prediction change| mismatched H":report["prediction_abs_change_on_hidden_mismatch"],
        "|router change| mismatched H":report["pop_weight_abs_change_on_hidden_mismatch"],
    }
    for label,v in fields.items():
        print(f"{label:34} {v['mean']:9.3f} {v['std']:9.3f} "
              f"{v['min']:9.3f} {v['max']:9.3f}")
    corr=report["teacher_student_pop_correlation"]
    print("[k7-m6-diag] teacher/student pop correlation="+
          (f"{corr:.3f}" if corr is not None else "undefined (constant)"))
    print(f"[k7-m6-diag] saved {output}")


if __name__=="__main__":
    main()
