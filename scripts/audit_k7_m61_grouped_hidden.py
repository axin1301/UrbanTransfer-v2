"""M6.1 raw grouped MLLM hidden-token audit (NO fits or MLLM forwards).

Source-training images only. Compare same-image task prompts at raw
visual/definition/relation token level, after Student projection, and
after learned cross-attention. Do not access target labels, pseudo-dev
images, or official test datasets.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.run_k7_m61_student import load_precomputed
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m61_router import StructuredMLLMRouter
from urban_transfer.k7_mllm_hidden import HiddenStore,TASKS
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK


def statistics(values):
    arr=np.asarray(values,dtype=np.float64).reshape(-1)
    if not arr.size or not np.isfinite(arr).all():
        raise ValueError("Nonfinite or empty MLLM diagnostic")
    return {"mean":float(arr.mean()),"std":float(arr.std()),
            "median":float(np.median(arr)),
            "min":float(arr.min()),"max":float(arr.max())}


def spans(counts):
    if len(counts)!=3 or any(int(n)<1 for n in counts):
        raise ValueError("Expected three positive token group sizes")
    a,b,c=map(int,counts)
    return {"visual":slice(0,a),"definition":slice(a,a+b),
            "relation":slice(a+b,a+b+c)}


def token_pair_metrics(a,b):
    """Matched-token cosine, per-image pooled cosine and relative delta."""
    if a.shape!=b.shape or a.ndim!=3:
        raise ValueError("Expected paired [N,tokens,dim] tensors")
    a=a.float()
    b=b.float()
    matched=F.cosine_similarity(a,b,dim=-1).mean(-1)
    pooled=F.cosine_similarity(a.mean(1),b.mean(1),dim=-1)
    delta=(a-b).flatten(1).norm(dim=-1)
    reference=.5*(a.flatten(1).norm(dim=-1)+b.flatten(1).norm(dim=-1))
    relative=delta/reference.clamp_min(1e-8)
    # Center the pair before cosine to see differences hidden by a
    # dominant shared baseline direction. Metrics are descriptive.
    center=.5*(a+b)
    centered_delta=(a-center).flatten(1).norm(dim=-1)
    return {"matched_token_cosine":statistics(matched.numpy()),
            "pooled_cosine":statistics(pooled.numpy()),
            "relative_frobenius_difference":statistics(relative.numpy()),
            "centered_pair_distance":statistics((2*centered_delta).numpy())}


@torch.inference_mode()
def analyze(cache,indices,model,counts,batch_size):
    parts=spans(counts)
    pairs=((TRAIN_TASKS[0],HELDOUT_TASK),
           (TRAIN_TASKS[1],HELDOUT_TASK),
           (TRAIN_TASKS[0],TRAIN_TASKS[1]))
    batches={t:[] for t in TASKS}
    projected={t:[] for t in TASKS}
    knowledge={t:[] for t in TASKS}
    weights={t:[] for t in TASKS}
    for start in range(0,len(indices),batch_size):
        ids=indices[start:start+batch_size]
        for task in TASKS:
            raw=torch.stack([cache[int(i),task] for i in ids]).float()
            batches[task].append(raw)
            # Project identical raw groups through the actual trained
            # M6.1 token projection, before segment embedding/attention.
            projected[task].append(model.token_projection(raw))
            knowledge[task].append(model.read_hidden(raw))
            weights[task].append(F.softmax(model.router(
                knowledge[task][-1]),dim=-1)[:,0])
    raw={t:torch.cat(v,dim=0) for t,v in batches.items()}
    proj={t:torch.cat(v,dim=0) for t,v in projected.items()}
    know={t:torch.cat(v,dim=0) for t,v in knowledge.items()}
    routes={t:torch.cat(v,dim=0).numpy() for t,v in weights.items()}
    report={}
    for left,right in pairs:
        name=left+"__vs__"+right
        detail={}
        for group,segment in parts.items():
            detail[group]={
                "raw":token_pair_metrics(raw[left][:,segment],
                                         raw[right][:,segment]),
                "student_projected":token_pair_metrics(
                    proj[left][:,segment],proj[right][:,segment]),
            }
        left_z=know[left]
        right_z=know[right]
        detail["student_cross_attended"]={
            "cosine":statistics(F.cosine_similarity(left_z,right_z,dim=-1).numpy()),
            "relative_l2":statistics(((left_z-right_z).norm(dim=-1)/
                (.5*(left_z.norm(dim=-1)+right_z.norm(dim=-1))).clamp_min(1e-8)).numpy()),
            "abs_population_weight_delta":statistics(np.abs(routes[left]-routes[right])),
        }
        report[name]=detail
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,
                   default=Path("runs/k7/m61_structured_seed42"))
    p.add_argument("--output",type=Path)
    p.add_argument("--batch-size",type=int,default=32)
    a=p.parse_args()
    if a.batch_size<1:p.error("Batch size must be positive")
    manifest=json.loads((a.run_dir/"manifest.json").read_text())
    config=json.loads((a.run_dir/"config.json").read_text())
    if (config["seed"]!=42 or tuple(config["source_tasks"])!=TRAIN_TASKS or
        config["heldout_task"]!=HELDOUT_TASK or
        list(config["group_counts"])!=list(manifest["group_counts"])):
        raise ValueError("M6.1 config and cache mismatch")
    indices=np.asarray(manifest["train"],dtype=np.int64)
    frame,features,_,_,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    if (not len(indices) or len(set(indices.tolist()))!=len(indices) or
        not np.isin(frame.iloc[indices].indicator_column.astype(str),
                    TRAIN_TASKS).all() or
        frame.iloc[indices].city.astype(str).isin(manifest["held_cities"]).any()):
        raise ValueError("Must audit only disjoint source training rows")
    ds=ImageUrbanDataset(manifest["source_csv"],manifest["image_root"],
                         partition="eval")
    if (len(ds.df)!=len(frame) or
        ds.df.indicator_column.astype(str).tolist()!=names):
        raise ValueError("Image rows do not match cached tensors")
    # Do not even access target validation image paths or labels.
    cache=load_precomputed(frame,indices,TASKS,ds,
                           HiddenStore(a.run_dir/"hidden.sqlite3"),manifest)
    d=cache[int(indices[0]),TASKS[0]].shape[-1]
    model=StructuredMLLMRouter(
        features.shape[-1],d,hidden=int(config["hidden"]),
        heads=int(config["heads"]),
        group_counts=tuple(manifest["group_counts"]))
    checkpoint=torch.load(a.run_dir/"model.pt",
                          map_location="cpu",weights_only=True)
    if checkpoint["heldout_cities"]!=manifest["held_cities"]:
        raise ValueError("Checkpoint city split mismatch")
    model.load_state_dict(checkpoint["model"],strict=True)
    model.eval()
    result=analyze(cache,indices,model,
                   manifest["group_counts"],a.batch_size)
    document={"protocol":"SOURCE TRAIN ONLY; no labels read/used for analysis; frozen MLLM/Student; no optimization",
              "n_source_rows":len(indices),
              "tokens_per_group":dict(zip(("visual","definition","relation"),
                                           manifest["group_counts"])),
              "hidden_dim":d,"pairs":result,
              "interpretation":"Very high cosine may hide small but real differences; compare relative distances at raw, projection and cross-attention stages."}
    output=a.output or a.run_dir/"raw_grouped_hidden_audit.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing report: "+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(document,indent=2))
    print(f"[k7-m61-groups] n={len(indices)} dim={d} groups={manifest['group_counts']}")
    for pair,groups in result.items():
        print("[k7-m61-groups]",pair)
        for part in ("visual","definition","relation"):
            source=groups[part]["raw"]
            student=groups[part]["student_projected"]
            print(f"  {part:11s} RAW cos={source['pooled_cosine']['mean']:.5f} "
                  f"rel_diff={source['relative_frobenius_difference']['mean']:.5f} "
                  f"| projected cos={student['pooled_cosine']['mean']:.5f} "
                  f"rel_diff={student['relative_frobenius_difference']['mean']:.5f}")
        final=groups["student_cross_attended"]
        print(f"  FUSED       cosine={final['cosine']['mean']:.6f} "
              f"rel_diff={final['relative_l2']['mean']:.6f} "
              f"router_abs_delta={final['abs_population_weight_delta']['mean']:.6f}")
    print(f"[k7-m61-groups] saved {output}")


if __name__=="__main__":
    main()
