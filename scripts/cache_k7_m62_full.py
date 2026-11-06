"""Resumable full-scale Qwen2.5-VL hidden-state caching, source-only.

--audit-only counts unique content keys and missing cache entries WITHOUT
loading Qwen-VL. Encoding writes to runs/k7/m62_full_seed42/hidden.sqlite3.
All keys include image path, metadata, frozen prompt, model and token config.
No official final test CSV is opened. No target Education labels used.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from tqdm.auto import tqdm

from scripts.prepare_k7_m62_full import build_job_index
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_mllm_hidden import HiddenStore
from urban_transfer.k7_m61_structured import (
    StructuredFrozenMLLM,structured_key)


def schedule_jobs(frame,manifest,ds):
    """Exactly one work item per unique image+task content-addressed key."""
    if len(ds.df)!=len(frame) or (
        ds.df.indicator_column.astype(str).tolist()!=
        frame.indicator_column.astype(str).tolist()):
        raise ValueError("Image source dataframe differs from tensor rows")
    jobs={}
    path_cache={}
    names=frame.indicator_column.astype(str).to_numpy()
    for idx,task in build_job_index(frame,manifest):
        if idx not in path_cache:
            path_cache[idx]=str(ds.find_image(ds.df.iloc[int(idx)]))
        image_path=path_cache[idx]
        key=structured_key(image_path,task,manifest["model_id"],
                           manifest["max_pixels"],manifest["group_counts"])
        if key not in jobs:
            jobs[key]=(image_path,task,key)
    return list(jobs.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,
                   default=Path("runs/k7/m62_full_seed42"))
    p.add_argument("--audit-only",action="store_true")
    p.add_argument("--reuse-sqlite",type=Path,nargs="*",default=[],
                   help="Read-only existing M6.1/UCSI SQLite caches; safe key-based reuse")
    a=p.parse_args()
    manifest=json.loads((a.run_dir/"full_manifest.json").read_text())
    if manifest["seed"]!=42:
        raise ValueError("Full benchmark requires seed42")
    frame,_,_,_,_=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    ds=ImageUrbanDataset(manifest["source_csv"],manifest["image_root"],
                         partition="eval")
    jobs=schedule_jobs(frame,manifest,ds)
    store=HiddenStore(a.run_dir/"hidden.sqlite3")
    prior=[]
    for old_path in a.reuse_sqlite:
        if not old_path.is_file():
            raise FileNotFoundError("Reuse cache absent: "+str(old_path))
        if old_path.resolve()==(a.run_dir/"hidden.sqlite3").resolve():
            raise ValueError("Do not reuse the destination DB as an external cache")
        prior.append(HiddenStore(old_path))
    missing=[]
    reusable=[]
    cached=0
    for image_path,task,key in tqdm(jobs,desc="Inspect Qwen hidden cache keys"):
        if store.get(key) is not None:
            cached+=1
            continue
        found=None
        for old in prior:
            found=old.get(key)
            if found is not None:
                break
        if found is None:
            missing.append((image_path,task,key))
        else:
            reusable.append((key,task,found))

    raw_gib=len(jobs)*sum(manifest["group_counts"])*2048*2/(1024**3)
    print(f"[k7-full-cache] requests={manifest['n_encoding_requests_before_dedup']} "
          f"unique_keys={len(jobs)} cached={cached} "
          f"reusable={len(reusable)} missing={len(missing)} raw_GiB={raw_gib:.2f}",flush=True)
    if a.audit_only:
        print("[k7-full-cache] AUDIT ONLY: no Qwen-VL model loaded",flush=True)
        return
    for key,task,found in tqdm(reusable,desc="Copy existing frozen hidden keys"):
        store.put(key,task,found)
    if missing:
        encoder=StructuredFrozenMLLM(
            manifest["model_id"],max_pixels=int(manifest["max_pixels"]))
        for image_path,task,key in tqdm(
            missing,desc="Full frozen Qwen2.5-VL",dynamic_ncols=True):
            state=encoder.encode(
                image_path,task,counts=tuple(manifest["group_counts"]))
            store.put(key,task,state)
    print(f"[k7-full-cache] COMPLETE {a.run_dir}",flush=True)


if __name__=="__main__":
    main()
