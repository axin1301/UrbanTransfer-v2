"""One-time UCSI MLLM encoding for the fixed M6.1 held-city split.

Only Population/Food rows in held cities, sampled from SOURCE train.csv.
Write to a NEW dedicated SQLite DB; never append to M6.1's original DB.
No labels are passed to Qwen2.5-VL. Resume partial cache safely.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm.auto import tqdm

from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS
from urban_transfer.k7_mllm_hidden import HiddenStore
from urban_transfer.k7_m61_structured import StructuredFrozenMLLM,structured_key
from scripts.run_k7_m62_task_query import validate_split


def select_ucsi(frame,base,limit=64,seed=42):
    """Deterministic held-city × SOURCE-task quota round robin."""
    if limit<2 or seed!=42:
        raise ValueError("Expected positive fixed seed42 UCSI budget")
    task=frame.indicator_column.astype(str)
    unseen=frame.city.astype(str).isin(base["held_cities"])
    eligible=frame.loc[unseen & task.isin(TRAIN_TASKS)].copy()
    blocked=set(base["train"]+base["SCUI"]+base["UCUI"])
    eligible=eligible.loc[~eligible.index.isin(blocked)]
    if eligible.empty or not set(TRAIN_TASKS).issubset(set(eligible.indicator_column)):
        raise ValueError("Missing held-city Population or Food rows")
    rng=np.random.default_rng(seed+1005)
    groups=[]
    for _,group in eligible.groupby(["city","indicator_column"],sort=True):
        ids=group.index.to_numpy(dtype=np.int64)
        groups.append(list(map(int,rng.permutation(ids))))
    selected=[]
    while len(selected)<min(limit,len(eligible)):
        progressed=False
        for group in groups:
            if group and len(selected)<limit:
                selected.append(group.pop())
                progressed=True
        if not progressed:break
    if len(selected)<2:
        raise ValueError("Insufficient source-held-city audit rows")
    picked=frame.iloc[selected]
    if (not picked.city.astype(str).isin(base["held_cities"]).all()
        or not picked.indicator_column.isin(TRAIN_TASKS).all()):
        raise ValueError("Source UCSI sample leaked outside held city/tasks")
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir",type=Path,
                   default=Path("runs/k7/m61_structured_seed42"))
    p.add_argument("--run-dir",type=Path,
                   default=Path("runs/k7/m62_fixed_comparison_seed42"))
    p.add_argument("--limit",type=int,default=64)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if a.run_dir.resolve()==a.cache_dir.resolve():
        raise ValueError("New UCSI DB must not be in M6.1 directory")
    base=json.loads((a.cache_dir/"manifest.json").read_text())
    frame,x,ctx,y,names=read_tensor_cache(
        Path(base["tensor_cache"]),Path(base["source_csv"]),
        Path(base["image_root"]),Path(base["feature_dir"]),
        base["remoteclip_model"])
    validate_split(frame,base)
    if a.seed!=42 or a.limit!=64:
        p.error("Pre-registered UCSI sample: 64, seed42")
    ids=select_ucsi(frame,base,a.limit,a.seed)
    a.run_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=a.run_dir/"ucsi_manifest.json"
    metadata={
        "protocol":"held-city source-task-only UCSI from M6.1 source train CSV",
        "base_manifest":str((a.cache_dir/"manifest.json").resolve()),
        "held_cities":base["held_cities"],
        "UCSI":ids,"seed":a.seed,"limit":a.limit,
        "source_tasks":list(TRAIN_TASKS),
        "model_id":base["model_id"],
        "max_pixels":base["max_pixels"],
        "group_counts":base["group_counts"]}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())!=metadata:
            raise ValueError("Refuse to change frozen UCSI manifest")
    else:
        manifest_path.write_text(json.dumps(metadata,indent=2))
    ds=ImageUrbanDataset(base["source_csv"],base["image_root"],
                         partition="eval")
    if len(ds.df)!=len(frame) or names!=ds.df.indicator_column.astype(str).tolist():
        raise ValueError("Shared source cache row-order mismatch")
    store=HiddenStore(a.run_dir/"ucsi_hidden.sqlite3")
    missing=[]
    for idx in ids:
        path=str(ds.find_image(ds.df.iloc[idx]))
        task=names[idx]
        key=structured_key(path,task,base["model_id"],
                           base["max_pixels"],base["group_counts"])
        if store.get(key) is None:
            missing.append((path,task,key))
    print(f"[k7-m62-ucsi-cache] UCSI={len(ids)} missing={len(missing)} "
          f"Pop={sum(names[i]==TRAIN_TASKS[0] for i in ids)} "
          f"Food={sum(names[i]==TRAIN_TASKS[1] for i in ids)}",flush=True)
    if missing:
        # Instantiate the heavy MLLM ONLY when needed.
        encoder=StructuredFrozenMLLM(
            base["model_id"],max_pixels=base["max_pixels"])
        for path,task,key in tqdm(missing,desc="UCSI source-task Qwen-VL cache"):
            h=encoder.encode(path,task,counts=tuple(base["group_counts"]))
            store.put(key,task,h)
    print(f"[k7-m62-ucsi-cache] COMPLETE {a.run_dir}",flush=True)


if __name__=="__main__":
    main()
