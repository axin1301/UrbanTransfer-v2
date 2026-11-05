"""K7 M6.1: one student fit on structured image/definition/relation hidden tokens.

Precompute with scripts/cache_k7_m61_structured.py first. Source train labels
only; Education labels are read only after optimization for SCUI/UCUI.
No official test files are imported.
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
from tqdm.auto import tqdm

from scripts.run_z1_pilot import calibrated_moments,scores
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.k7_mllm_hidden import HiddenStore,TASKS
from urban_transfer.k7_m61_structured import structured_key
from urban_transfer.k7_m61_router import StructuredMLLMRouter
from urban_transfer.ktrans_text import qwen_task_vectors


def load_precomputed(frame,indices,tasks,ds,store,manifest):
    # Loads precisely the selected image/task cached hidden arrays.
    # No MLLM inference or target label access here.
    path_by_id={}
    result={}
    for idx in indices:
        idx=int(idx)
        if idx not in path_by_id:
            path_by_id[idx]=str(ds.find_image(ds.df.iloc[idx]))
        for task in tasks:
            key=structured_key(path_by_id[idx],task,manifest["model_id"],
                               manifest["max_pixels"],manifest["group_counts"])
            value=store.get(key)
            if value is None:
                raise FileNotFoundError(
                    "Missing MLLM hidden cache; rerun cache script: "+key)
            if value.ndim!=2 or value.shape[0]!=sum(manifest["group_counts"]):
                raise ValueError("MLLM hidden cache token mismatch")
            result[(idx,task)]=value
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,
                   default=Path("runs/k7/m61_structured_seed42"))
    p.add_argument("--text-cache-dir",type=Path,default=Path("runs/ktrans_text"))
    p.add_argument("--text-model",default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--hidden",type=int,default=128)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--target-align-weight",type=float,default=.25)
    p.add_argument("--self-loss",type=float,default=.3)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M6.1 student requires MI250/ROCm GPU")
    if a.seed!=42 or a.epochs<1 or a.batch_size<1 or a.target_align_weight<=0 or a.self_loss<0:
        p.error("Fixed seed42, positive budget and alignment strength required")
    if (a.run_dir/"model.pt").exists() or (a.run_dir/"metrics.csv").exists():
        raise FileExistsError("M6.1 training results already exist; do not overwrite")
    manifest_path=a.run_dir/"manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Precompute MLLM cache/manifest before student training")
    manifest=json.loads(manifest_path.read_text())
    if tuple(manifest.get("held_cities",[]))==():
        raise ValueError("No held cities in M6.1 manifest")
    frame,x,ctx,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    train_ids=np.asarray(manifest["train"],dtype=np.int64)
    scui=np.asarray(manifest["SCUI"],dtype=np.int64)
    ucui=np.asarray(manifest["UCUI"],dtype=np.int64)
    unseen=frame.city.astype(str).isin(manifest["held_cities"]).to_numpy()
    tasks=frame.indicator_column.astype(str).to_numpy()
    if not (np.isin(tasks[train_ids],TRAIN_TASKS).all() and
            not unseen[train_ids].any() and
            np.all(tasks[scui]==HELDOUT_TASK) and
            np.all(tasks[ucui]==HELDOUT_TASK) and
            not unseen[scui].any() and unseen[ucui].all()):
        raise ValueError("M6.1 manifest violates source-only target holdout policy")
    ds=ImageUrbanDataset(manifest["source_csv"],manifest["image_root"],
                         partition="eval")
    if (len(ds.df)!=len(frame) or
        ds.df.indicator_column.astype(str).tolist()!=names):
        raise ValueError("Image manifest/cache row mismatch")
    store=HiddenStore(a.run_dir/"hidden.sqlite3")
    train_cache=load_precomputed(
        frame,train_ids,TASKS,ds,store,manifest)
    eval_ids=np.unique(np.concatenate((scui,ucui)))
    evaluation_cache=load_precomputed(
        frame,eval_ids,(HELDOUT_TASK,),ds,store,manifest)
    embeddings=qwen_task_vectors(root=a.text_cache_dir,model_id=a.text_model,
                                  profile="short",only_cached=True)
    train_frame=frame.iloc[train_ids]
    if set(train_frame.indicator_column)!=set(TRAIN_TASKS):
        raise ValueError("M6.1 pilot must train on Population and Food")
    stats,unseen_stats=calibrated_moments(
        train_frame,embeddings,HELDOUT_TASK,temperature=.2)
    ids=torch.from_numpy(train_ids.copy())
    src_names=[names[int(i)] for i in train_ids]
    labels=torch.tensor([(float(y[int(i)])-stats[name][0])/stats[name][1]
                         for i,name in zip(train_ids,src_names)],dtype=torch.float32)
    supervised_tokens=torch.stack([train_cache[int(i),name]
                                  for i,name in zip(train_ids,src_names)])
    target_tokens=torch.stack([train_cache[int(i),HELDOUT_TASK]
                               for i in train_ids])
    val_tokens={int(i):evaluation_cache[int(i),HELDOUT_TASK]
                for i in eval_ids}
    device=torch.device("cuda")
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    model=StructuredMLLMRouter(
        x.shape[1],supervised_tokens.shape[-1],
        hidden=a.hidden,heads=a.heads,
        group_counts=tuple(manifest["group_counts"])).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    visual=x[ids].to(device)
    context=ctx[ids].to(device)
    source_h=supervised_tokens.to(device)
    target_h=target_tokens.to(device)
    targets=labels.to(device)
    source_indices=torch.tensor(
        [TRAIN_TASKS.index(n) for n in src_names],
        device=device,dtype=torch.long)
    generator=torch.Generator(device="cpu").manual_seed(a.seed)
    print(f"[k7-m61] GPU={torch.cuda.get_device_name(0)} "
          f"train={len(train_ids)} SCUI={len(scui)} UCUI={len(ucui)} "
          f"MLLM_hidden_tokens={source_h.shape[1]} dim={source_h.shape[2]} "
          "seed=42 trained_models=1",flush=True)
    for epoch in tqdm(range(a.epochs),desc="M6.1 structured hidden router single fit",
                      dynamic_ncols=True):
        model.train()
        order=torch.randperm(len(train_ids),generator=generator).to(device)
        total=0.
        steps=0
        for first in range(0,len(order),a.batch_size):
            j=order[first:first+a.batch_size]
            opt.zero_grad(set_to_none=True)
            # Both expert routes participate in source regression.
            pred=model(visual[j],context[j],source_h[j])
            supervised=F.mse_loss(pred,targets[j])
            # Auxiliary direct training ensures each source expert has a
            # meaningful independent specialization, with no main mask.
            source_pred=model.source_expert_predictions(
                visual[j],context[j],source_h[j])
            direct=source_pred.gather(1,source_indices[j].unsqueeze(1)).squeeze(1)
            self_loss=F.mse_loss(direct,targets[j])
            # Education definition participates in training on SOURCE tiles;
            # frozen RemoteCLIP image features act as image-only anchors.
            # No Education target label or inference image is involved.
            knowledge=model.read_hidden(target_h[j])
            align=model.text_image_alignment(knowledge,visual[j])
            loss=supervised+a.self_loss*self_loss+a.target_align_weight*align
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step()
            total+=float(loss.detach().cpu())
            steps+=1
        if epoch==0 or epoch+1==a.epochs or (epoch+1)%5==0:
            print(f"[k7-m61] epoch={epoch+1}/{a.epochs} "
                  f"loss={total/max(steps,1):.3f}",flush=True)
    model.eval()
    results=[]
    for split,subset in (("SCUI",scui),("UCUI",ucui)):
        out,weights=[],[]
        with torch.inference_mode():
            for first in range(0,len(subset),a.batch_size):
                ix=subset[first:first+a.batch_size]
                toks=torch.stack([val_tokens[int(i)] for i in ix]).to(device)
                z,w,_=model(x[ix].to(device),ctx[ix].to(device),toks,
                            return_aux=True)
                out.append(z.cpu())
                weights.append(w.cpu())
        pred=unseen_stats[0]+unseen_stats[1]*torch.cat(out).numpy()
        truth=y[torch.from_numpy(subset.copy())].numpy()
        metric=scores(pred,truth)
        baseline=float(np.mean((unseen_stats[0]-truth)**2))
        attention=torch.cat(weights).mean(0).numpy()
        results.append({"partition":split,**metric,
                        "relative_mse_to_source_mean":metric["mse_log1p"]/max(baseline,1e-12),
                        "population_weight":float(attention[0]),
                        "food_weight":float(attention[1])})
    a.run_dir.mkdir(parents=True,exist_ok=True)
    with (a.run_dir/"metrics.csv").open("w",newline="") as fd:
        writer=csv.DictWriter(fd,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    torch.save({"model":model.cpu().state_dict(),"train_stats":stats,
                "unseen_stats":unseen_stats,
                "heldout_cities":manifest["held_cities"]},
               a.run_dir/"model.pt")
    (a.run_dir/"config.json").write_text(json.dumps({
        **{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        "source_tasks":TRAIN_TASKS,"heldout_task":HELDOUT_TASK,
        "n_train":len(train_ids),"n_scui":len(scui),"n_ucui":len(ucui),
        "protocol":"source two-expert regression; Education definition text-only unlabeled image-contrastive alignment on source tiles; Education labels only scored",
        "group_counts":manifest["group_counts"]
    },indent=2))
    for row in results:
        print(f"[k7-m61] {row['partition']} MSE={row['mse_log1p']:.3f} "
              f"Pearson={row['pearson']:.3f} "
              f"Ratio={row['relative_mse_to_source_mean']:.3f} "
              f"PopWt={row['population_weight']:.3f} "
              f"FoodWt={row['food_weight']:.3f}",flush=True)
    print(f"[k7-m61] COMPLETE {a.run_dir}",flush=True)


if __name__=="__main__":
    main()
