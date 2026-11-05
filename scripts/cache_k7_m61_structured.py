"""M6.1 structured visual/definition/relation MLLM hidden cache.

Subset is deliberately smaller than M3/M4 to avoid unbounded 3B MLLM
inference. Source task train rows are sampled by city×indicator. Education
evaluation rows are sampled by city; their labels are never used here.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm.auto import tqdm
from urban_transfer.data_m1 import ImageUrbanDataset,stratified_sample
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.k7_mllm_hidden import TASKS,HiddenStore
from urban_transfer.k7_m61_structured import StructuredFrozenMLLM,structured_key


def selection(frame,seed,train_limit,eval_limit,city_fraction):
    if seed!=42 or train_limit<2 or eval_limit<1:
        raise ValueError("Fixed seed=42, positive train/eval limits required")
    cities=sorted(frame.city.astype(str).unique())
    if len(cities)<3 or not 0<city_fraction<.5:
        raise ValueError("Invalid city partition")
    count=max(1,min(len(cities)-2,round(len(cities)*city_fraction)))
    held=sorted(np.random.default_rng(seed).permutation(cities)[:count].tolist())
    unseen=frame.city.astype(str).isin(held).to_numpy()
    task=frame.indicator_column.astype(str)
    source=task.isin(TRAIN_TASKS).to_numpy()
    target=task.eq(HELDOUT_TASK).to_numpy()
    train=np.flatnonzero(source & ~unseen)
    scui=np.flatnonzero(target & ~unseen)
    ucui=np.flatnonzero(target & unseen)
    if not min(len(train),len(scui),len(ucui)):
        raise ValueError("Empty MLLM pilot fold")
    def choose(ids,limit,offset):
        if len(ids)<=limit:return ids
        chunk=frame.iloc[ids].copy().reset_index()
        # Stratify by city for evaluation, city and source task for training.
        if offset:
            rng=np.random.default_rng(seed+offset)
            grouped=[rng.permutation(g).tolist() for _,g in
                     chunk.groupby(["city"],sort=True).indices.items()]
            chosen=[]
            while len(chosen)<limit:
                any_item=False
                for group in grouped:
                    if group and len(chosen)<limit:
                        chosen.append(group.pop())
                        any_item=True
                if not any_item:break
            return chunk.iloc[chosen]["index"].to_numpy(dtype=np.int64)
        result=stratified_sample(chunk,limit,seed)
        return result["index"].to_numpy(dtype=np.int64)
    return {"held_cities":held,
            "train":choose(train,train_limit,0).tolist(),
            "SCUI":choose(scui,eval_limit,1).tolist(),
            "UCUI":choose(ucui,eval_limit,2).tolist()}


def jobs(frame,manifest):
    # Every source-supervised row receives BOTH source tasks and held-out
    # Education prompt. This explicitly exposes target text to training.
    # Evaluation receives only Education prompt, never source labels.
    work=[]
    for idx in manifest["train"]:
        work.extend((idx,task) for task in TASKS)
    for part in ("SCUI","UCUI"):
        work.extend((idx,HELDOUT_TASK) for idx in manifest[part])
    return work


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv",type=Path,required=True)
    p.add_argument("--image-root",type=Path,required=True)
    p.add_argument("--feature-dir",type=Path,default=Path("runs/m3_features"))
    p.add_argument("--tensor-cache",type=Path,default=Path("runs/k7/e1_e2_v1/_source_tensors.pt"))
    p.add_argument("--remoteclip-model",default="ViT-B-32")
    p.add_argument("--model-id",default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--run-dir",type=Path,default=Path("runs/k7/m61_structured_seed42"))
    p.add_argument("--train-limit",type=int,default=256)
    p.add_argument("--eval-limit",type=int,default=128)
    p.add_argument("--city-fraction",type=float,default=.25)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--visual-tokens",type=int,default=32)
    p.add_argument("--definition-tokens",type=int,default=16)
    p.add_argument("--relation-tokens",type=int,default=16)
    p.add_argument("--max-pixels",type=int,default=100352)
    a=p.parse_args()
    if min(a.visual_tokens,a.definition_tokens,a.relation_tokens)<1 or a.max_pixels<56*56:
        p.error("Invalid MLLM tokens/image resolution")
    frame,_,_,_,names=read_tensor_cache(
        a.tensor_cache,a.source_csv,a.image_root,a.feature_dir,a.remoteclip_model)
    manifest_path=a.run_dir/"manifest.json"
    proposed=selection(frame,a.seed,a.train_limit,a.eval_limit,a.city_fraction)
    metadata={**proposed,"model_id":a.model_id,"group_counts":[a.visual_tokens,a.definition_tokens,a.relation_tokens],
              "max_pixels":a.max_pixels,"source_csv":str(a.source_csv.resolve()),
              "image_root":str(a.image_root.resolve()),
              "tensor_cache":str(a.tensor_cache.resolve()),
              "feature_dir":str(a.feature_dir.resolve()),
              "remoteclip_model":a.remoteclip_model}
    a.run_dir.mkdir(parents=True,exist_ok=True)
    if manifest_path.exists():
        saved=json.loads(manifest_path.read_text())
        if saved!=metadata:
            raise ValueError("Existing MLLM manifest mismatch; use a NEW run-dir")
    else:
        manifest_path.write_text(json.dumps(metadata,indent=2))
    store=HiddenStore(a.run_dir/"hidden.sqlite3")
    ds=ImageUrbanDataset(a.source_csv,a.image_root,partition="eval")
    if len(ds.df)!=len(frame) or ds.df.indicator_column.astype(str).tolist()!=names:
        raise ValueError("MLLM image row order differs from shared tensor cache")
    tasks=jobs(frame,metadata)
    # Only resolve image paths for the selected tiny MLLM pilot.
    path_by_index={}
    for index in sorted({i for i,_ in tasks}):
        path_by_index[index]=str(ds.find_image(ds.df.iloc[index]))
    seen=set()
    missing=[]
    for index,task in tasks:
        path=path_by_index[index]
        key=structured_key(path,task,a.model_id,a.max_pixels,metadata["group_counts"])
        if key in seen:continue
        seen.add(key)
        if store.get(key) is None:missing.append((path,task,key))
    print(f"[k7-m61-cache] train={len(metadata['train'])} "
          f"SCUI={len(metadata['SCUI'])} UCUI={len(metadata['UCUI'])} "
          f"unique_requests={len(seen)} missing={len(missing)}",flush=True)
    if not missing:
        print(f"[k7-m61-cache] COMPLETE {a.run_dir}",flush=True)
        return
    encoder=StructuredFrozenMLLM(a.model_id,max_pixels=a.max_pixels)
    for path,task,key in tqdm(missing,desc="Frozen MLLM hidden states",
                              dynamic_ncols=True):
        hidden=encoder.encode(path,task,counts=tuple(metadata["group_counts"]))
        store.put(key,task,hidden)
    print(f"[k7-m61-cache] COMPLETE {a.run_dir}",flush=True)


if __name__=="__main__":
    main()
