"""Full-data M6.2 manifest from SOURCE train.csv ONLY.

Do not accept official test CSV paths. Existing shared RemoteCLIP tensor
cache is source-only. Split held cities deterministically, reserve a
small seen-city source validation subset for checkpoint selection (NOT
Education); keep all other eligible source rows for fitting.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK


def split_full(frame,seed=42,city_fraction=.25,source_val_fraction=.10):
    if seed!=42 or not (0<city_fraction<.5) or not (0<source_val_fraction<.5):
        raise ValueError("Pre-registered seed42 and nontrivial fractions required")
    cities=sorted(frame.city.astype(str).unique())
    if len(cities)<3:raise ValueError("Need at least 3 cities")
    rng=np.random.default_rng(seed)
    n_held=max(1,min(len(cities)-2,round(len(cities)*city_fraction)))
    held=sorted(rng.permutation(cities)[:n_held].tolist())
    unseen=frame.city.astype(str).isin(held).to_numpy()
    task=frame.indicator_column.astype(str).to_numpy()
    source=np.isin(task,TRAIN_TASKS)
    education=task==HELDOUT_TASK
    candidates=np.flatnonzero(source & ~unseen)
    if not len(candidates) or not (source & unseen).any():
        raise ValueError("Missing seen/held city source tasks")
    # Independent SOURCE-only validation by city×source-indicator strata.
    # The validation sample isn't used for any optimization; stratified
    # within seen cities, not for Education/model selection.
    val=[]
    train=[]
    for (_,indicator),group in frame.iloc[candidates].groupby(
        ["city","indicator_column"],sort=True):
        original=group.index.to_numpy(dtype=np.int64)
        shuffled=np.random.default_rng(
            seed+int(hashlib.sha256(
                (str(group.iloc[0].city)+"|"+indicator).encode()
            ).hexdigest()[:8],16)).permutation(original)
        n=max(1,int(round(len(shuffled)*source_val_fraction)))
        if n>=len(shuffled):
            raise ValueError("Source city×indicator stratum too small for train/val")
        val.extend(shuffled[:n].tolist())
        train.extend(shuffled[n:].tolist())
    groups={
        "train":sorted(train),
        "source_val":sorted(val),
        "SCUI":np.flatnonzero(education & ~unseen).tolist(),
        "UCSI":np.flatnonzero(source & unseen).tolist(),
        "UCUI":np.flatnonzero(education & unseen).tolist(),
    }
    all_indices=[i for v in groups.values() for i in v]
    if len(all_indices)!=len(set(all_indices)) or any(not v for v in groups.values()):
        raise ValueError("Empty or overlapping full-data partitions")
    for name in TRAIN_TASKS:
        if min(sum(task[np.asarray(groups[g])]==name)
               for g in ("train","source_val","UCSI"))<2:
            raise ValueError("Insufficient source indicator rows: "+name)
    return {"seed":seed,"city_fraction":city_fraction,
            "source_val_fraction":source_val_fraction,
            "held_cities":held,**groups}


def build_job_index(frame,manifest):
    """Cache unique image+prompt jobs, counting duplicates by key later."""
    task=frame.indicator_column.astype(str).to_numpy()
    jobs=[]
    for index in manifest["train"]:
        # Education description on SOURCE images for unlabeled alignment.
        jobs.extend((index,t) for t in (*TRAIN_TASKS,HELDOUT_TASK))
    for index in manifest["source_val"]:
        jobs.append((index,task[index]))
    for split in ("SCUI","UCSI","UCUI"):
        jobs.extend((index,task[index]) for index in manifest[split])
    return jobs


def manifest_main(args):
    output=args.run_dir/"full_manifest.json"
    if output.exists():
        raise FileExistsError("Refusing to regenerate an existing full manifest: "+str(output))
    # Validate the exact source-only cache signature; no official test.
    frame,_,_,_,names=read_tensor_cache(
        args.tensor_cache,args.train_csv,args.image_root,args.feature_dir,
        args.remoteclip_model)
    if set(names)!=set((*TRAIN_TASKS,HELDOUT_TASK)):
        raise ValueError("Expected Population/Food/Education-only source study")
    split=split_full(frame,args.seed,args.city_fraction,args.source_val_fraction)
    jobs=build_job_index(frame,split)
    # Approximate source image counts only. Paths and real hash keys
    # are computed by the caching script and precisely deduplicated.
    row_identity=frame.city.astype(str)+"|"+frame.image_name.astype(str)
    data={
        **split,
        "source_csv":str(args.train_csv.resolve()),
        "image_root":str(args.image_root.resolve()),
        "tensor_cache":str(args.tensor_cache.resolve()),
        "feature_dir":str(args.feature_dir.resolve()),
        "remoteclip_model":args.remoteclip_model,
        "model_id":args.model_id,
        "max_pixels":args.max_pixels,
        "group_counts":[32,16,16],
        "protocol":"FULL SOURCE-CSV ONLY. Official final test files excluded. Education labels never used in optimization or checkpoint selection.",
        "n_rows":len(frame),
        "n_encoding_requests_before_dedup":len(jobs),
        "estimated_unique_images_by_city_and_name":int(row_identity.nunique()),
        "estimated_raw_hidden_gib_without_dedup":
            len(jobs)*64*2048*2/(1024**3),
    }
    args.run_dir.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,indent=2))
    print(f"[k7-full] source_rows={len(frame)} held_cities={len(split['held_cities'])} "
          f"train={len(split['train'])} source_val={len(split['source_val'])} "
          f"SCUI={len(split['SCUI'])} UCSI={len(split['UCSI'])} "
          f"UCUI={len(split['UCUI'])}",flush=True)
    print(f"[k7-full] pre_dedup_MLLM_requests={len(jobs)} "
          f"approx_raw_GiB={data['estimated_raw_hidden_gib_without_dedup']:.2f} "
          f"estimated_unique_images={data['estimated_unique_images_by_city_and_name']}",flush=True)
    print(f"[k7-full] manifest={output}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv",type=Path,required=True)
    p.add_argument("--image-root",type=Path,required=True)
    p.add_argument("--feature-dir",type=Path,default=Path("runs/m3_features"))
    p.add_argument("--tensor-cache",type=Path,
                   default=Path("runs/k7/e1_e2_v1/_source_tensors.pt"))
    p.add_argument("--remoteclip-model",default="ViT-B-32")
    p.add_argument("--model-id",default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--max-pixels",type=int,default=100352)
    p.add_argument("--run-dir",type=Path,
                   default=Path("runs/k7/m62_full_seed42"))
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--city-fraction",type=float,default=.25)
    p.add_argument("--source-val-fraction",type=float,default=.10)
    a=p.parse_args()
    manifest_main(a)


if __name__=="__main__":
    main()
