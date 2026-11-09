"""M6.6: one-command paired source-city training of neighbor × indicator links.

Two 20-epoch independent fits with identical initial trainable weights:
 cross_indicator: target A attends (neighbor j, source/held task B) tokens
 spatial_only: target A attends neighbor j with only target A semantics.
Same architecture/optimizer/data/batch sequence/frozen visual anchor.
No Education labels or official test files used. Frozen Qwen prompt
prototypes for all three indicator descriptions are derived from 12
source-training images, with NO labels.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.run_k7_m62_full import (
    CachedTokens,validate_full,source_train_targets)
from scripts.run_k7_m63_full import (
    PersistentReadOnlyHiddenStore,load_training_tokens)
from scripts.run_k7_m65_dev_intervention import validate_dev_provenance
from scripts.run_k7_m65_robustness_suite import write_csv
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.k7_m65_neighborhood import load_or_build_neighborhood
from urban_transfer.k7_m66_indicator_graph import cached_physical_neighbor_graph
from urban_transfer.k7_m66_cross_indicator import NeighborIndicatorInteraction


ARMS=("cross_indicator","spatial_only")
TASKS=(*TRAIN_TASKS,HELDOUT_TASK)


def source_prompt_prototypes(cache,source_train_ids,group_counts):
    """Label-free prototypes from source training IMAGE prompts only."""
    ids=np.asarray(source_train_ids,dtype=np.int64)
    sampled=ids[np.linspace(0,len(ids)-1,num=min(12,len(ids)),
                            dtype=np.int64)]
    prototypes=[]
    for name in TASKS:
        tokens=cache.batch(sampled,[name]*len(sampled),
                           torch.device("cpu")).float()
        image_len,definition_len,relation_len=group_counts
        d=tokens[:,image_len:image_len+definition_len].mean((0,1))
        r=tokens[:,image_len+definition_len:
                 image_len+definition_len+relation_len].mean((0,1))
        prototypes.append(torch.cat((d,r),dim=0))
    return torch.stack(prototypes).contiguous()


def batch_features(ids,visual,geo,neighborhood,graph,device):
    node,dist,valid=graph
    if not bool(valid[ids].any(dim=1).all()):
        raise ValueError("Some source row has no physical neighbor; inspect source city")
    return (visual[ids].to(device),
            geo[ids].to(device),
            neighborhood[ids].to(device),
            visual[node[ids]].to(device),
            dist[ids].to(device),
            valid[ids].to(device))


def train_arm(model,arm,visual,geo,neighborhood,graph,y,names,
              train_ids,stats,cache,output,signature,args,device):
    checkpoint=output/"checkpoint_last.pt"
    target=source_train_targets(train_ids,names,y,stats)
    labels=torch.tensor([TRAIN_TASKS.index(names[int(i)])
                         for i in train_ids],dtype=torch.long)
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=.001,weight_decay=1e-4)
    rng=torch.Generator().manual_seed(42)
    first=0
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if saved.get("version")!=1 or saved.get("signature")!=signature:
            raise ValueError("M6.6 checkpoint protocol mismatch")
        model.load_state_dict(saved["model"],strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["shuffle_rng"])
        torch.set_rng_state(saved["cpu_rng"])
        if device.type=="cuda":
            torch.cuda.set_rng_state(saved["gpu_rng"],device=device)
        first=int(saved["completed_epoch"])
        if not 1<=first<=20:
            raise ValueError("M6.6 invalid resumed epoch")
        print(f"[m66] {arm} RESUME epoch={first}/20",flush=True)
    for epoch in range(first,20):
        model.train()
        model.base.eval()
        start_time=time.monotonic()
        permutation=torch.randperm(len(train_ids),generator=rng)
        losses=[]
        for start in range(0,len(permutation),args.batch_size):
            chosen=permutation[start:start+args.batch_size]
            ids=train_ids[chosen.numpy()]
            v,g,nb,nv,nd,nm=batch_features(
                ids,visual,geo,neighborhood,graph,device)
            source_h,pop_h,food_h,edu_h=load_training_tokens(
                cache,ids,names,device)
            yy=target[chosen].to(device)
            task_id=labels[chosen].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred,w,task,fused,base,delta=model(
                v,g,source_h,nb,nv,nd,nm,return_aux=True)
            own=model.source_expert_predictions(
                v,g,source_h,nb,nv,nd,nm).gather(
                    1,task_id[:,None]).squeeze(1)
            sep=model.source_task_separation(
                v,g,pop_h,food_h,nb,nv,nd,nm)
            align=model.target_image_alignment(
                v,g,edu_h,nb,nv,nd,nm)
            loss=(F.mse_loss(pred,yy)+.3*F.mse_loss(own,yy)+
                  .2*sep+.05*align+.02*delta.square().mean())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite M6.6 source loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],2.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if (start//args.batch_size+1)%50==0:
                print(f"[m66] {arm} epoch={epoch+1}/20 "
                      f"batch={start//args.batch_size+1} "
                      f"elapsed_s={time.monotonic()-start_time:.1f}",
                      flush=True)
        saved={
            "version":1,"signature":signature,"completed_epoch":epoch+1,
            "model":{k:v.detach().cpu().clone()
                     for k,v in model.state_dict().items()},
            "optimizer":optimizer.state_dict(),
            "shuffle_rng":rng.get_state(),
            "cpu_rng":torch.get_rng_state()}
        if device.type=="cuda":
            saved["gpu_rng"]=torch.cuda.get_rng_state(device=device)
        temp=checkpoint.with_name(checkpoint.name+".tmp")
        torch.save(saved,temp)
        os.replace(temp,checkpoint)
        print(f"[m66] {arm} epoch={epoch+1}/20 "
              f"source_loss={np.mean(losses):.3f} "
              f"elapsed_s={time.monotonic()-start_time:.1f} "
              "checkpoint_saved=True",flush=True)


@torch.inference_mode()
def eval_arm(model,task,ids,visual,geo,neighborhood,graph,
             y,names,stats,cache,batch_size,device):
    model.eval()
    preds=[]; route=[]
    for start in range(0,len(ids),batch_size):
        ix=ids[start:start+batch_size]
        v,g,nb,nv,nd,nm=batch_features(
            ix,visual,geo,neighborhood,graph,device)
        tokens=cache.batch(ix,[names[int(i)] for i in ix],device)
        z,w,*_=model(v,g,tokens,nb,nv,nd,nm,return_aux=True)
        preds.append(z.detach().cpu().numpy())
        route.append(w[:,0].detach().cpu().numpy())
    normalized=np.concatenate(preds)
    pred=(stats[task][0]+stats[task][1]*normalized).astype(np.float64)
    truth=y[torch.as_tensor(ids,dtype=torch.long)].numpy().astype(np.float64)
    diff=pred-truth
    pearson=(float(np.corrcoef(pred,truth)[0,1])
             if min(pred.std(),truth.std())>1e-10 else float("nan"))
    return {
        "task":task,"n_rows":len(ids),
        "mse_log1p":float(np.mean(diff**2)),
        "mae_log1p":float(np.mean(np.abs(diff))),
        "pearson":pearson,"bias":float(diff.mean()),
        "pred_std":float(pred.std()),"truth_std":float(truth.std()),
        "population_weight":float(np.concatenate(route).mean())
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-dir",type=Path,
                   default=Path("runs/k7/m62_full_seed42"))
    p.add_argument("--dev-dir",type=Path,
                   default=Path("runs/k7/m65_neighborhood_dev_seed42"))
    p.add_argument("--neighbor-cache",type=Path,
                   default=Path("runs/k7/m65_neighbors_k8.pt"))
    p.add_argument("--graph-cache",type=Path,
                   default=Path("runs/k7/m66_physical_graph_k8.pt"))
    p.add_argument("--output",type=Path,
                   default=Path("runs/k7/m66_cross_indicator_dev_seed42"))
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=p.parse_args()
    if args.batch_size!=256 or args.epochs!=20:
        p.error("M6.6 fixed source-only protocol: 20 epochs, batch256")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    if args.output.resolve()==args.dev_dir.resolve():
        raise ValueError("Never overwrite M6.5 Dev checkpoint")
    if (args.output/"SUMMARY.csv").is_file():
        print(f"[m66] ALREADY COMPLETE {args.output}",flush=True)
        return

    manifest_path=(args.full_dir/"full_manifest.json").resolve()
    manifest=json.loads(manifest_path.read_text())
    frame,visual,geo,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    splits=validate_full(frame,manifest)
    _,train_ids,held_ids,cities,stats=validate_dev_provenance(
        args.dev_dir,manifest_path,frame,splits,y,names)
    neighborhood=load_or_build_neighborhood(
        frame,visual,manifest_path,args.neighbor_cache,k=8)
    graph=cached_physical_neighbor_graph(
        frame,manifest_path,args.graph_cache,k=8)
    anchor_path=args.dev_dir/"dev_visual_anchor.pt"
    anchor=torch.load(anchor_path,map_location="cpu",weights_only=True)
    output_config={
        "protocol":"m66_source_city_cross_indicator_pair_v1",
        "manifest":str(manifest_path),
        "dev_anchor_sha256":hashlib.sha256(
            anchor_path.read_bytes()).hexdigest(),
        "dev_train_sha256":hashlib.sha256(
            np.asarray(train_ids,dtype=np.int64).tobytes()).hexdigest(),
        "dev_held_sha256":hashlib.sha256(
            np.asarray(held_ids,dtype=np.int64).tobytes()).hexdigest(),
        "neighbor_cache":str(args.neighbor_cache.resolve()),
        "graph_cache":str(args.graph_cache.resolve()),
        "batch_size":256,"epochs":20,"seed":42,
        "arms":list(ARMS),"cities":cities,
        "semantic_prompt_images":"12 indices sampled only from source-train",
        "label_free_heldout_task_semantics":True,
        "education_label_usage":False,"official_test_usage":False,
    }
    args.output.mkdir(parents=True,exist_ok=True)
    cfg=args.output/"config.json"
    if cfg.exists():
        if json.loads(cfg.read_text())!=output_config:
            raise ValueError("Existing M6.6 experiment signature mismatch")
    else:
        if any(args.output.iterdir()):
            raise FileExistsError("Unrecognized M6.6 output directory")
        cfg.write_text(json.dumps(output_config,indent=2))
    ds=ImageUrbanDataset(manifest["source_csv"],manifest["image_root"],
                         partition="eval")
    if len(ds.df)!=len(frame) or names!=ds.df.indicator_column.astype(str).tolist():
        raise ValueError("M6.6 tensor/token alignment mismatch")
    store=PersistentReadOnlyHiddenStore(args.full_dir/"hidden.sqlite3")
    try:
        cache=CachedTokens(frame,ds,manifest,store,maxsize=512)
        semantic=source_prompt_prototypes(
            cache,train_ids,tuple(manifest["group_counts"]))
        token=cache[int(train_ids[0]),names[int(train_ids[0])]]
        device=torch.device(args.device)
        torch.manual_seed(42)
        if device.type=="cuda":
            torch.cuda.manual_seed_all(42)
        common=NeighborIndicatorInteraction(
            visual.shape[-1],token.shape[-1],semantic,
            mode=ARMS[0],hidden=128,heads=4,
            group_counts=tuple(manifest["group_counts"]))
        common.load_visual_baseline(anchor["model"])
        state={k:v.detach().cpu().clone()
               for k,v in common.state_dict().items()}
        cpu_rng=torch.get_rng_state().clone()
        gpu_rng=(torch.cuda.get_rng_state(device=device).clone()
                 if device.type=="cuda" else None)
        del common
        metrics=[]
        for arm in ARMS:
            directory=args.output/arm
            complete=directory/"metrics.csv"
            signature={"arm":arm,**output_config}
            if complete.is_file():
                checkpoint=directory/"checkpoint_last.pt"
                if not checkpoint.is_file():
                    raise ValueError("Existing M6.6 metrics without checkpoint")
                saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
                if saved["signature"]!=signature or saved["completed_epoch"]!=20:
                    raise ValueError("Existing completed M6.6 arm mismatch")
                with complete.open(newline="") as fh:
                    rows=list(csv.DictReader(fh))
                if {row["task"] for row in rows}!=set(TRAIN_TASKS):
                    raise ValueError("Corrupt completed M6.6 metrics")
                metrics.extend(rows)
                print(f"[m66] HIT completed {arm}",flush=True)
                continue
            directory.mkdir(parents=True,exist_ok=True)
            torch.set_rng_state(cpu_rng.clone())
            if gpu_rng is not None:
                torch.cuda.set_rng_state(gpu_rng.clone(),device=device)
            model=NeighborIndicatorInteraction(
                visual.shape[-1],token.shape[-1],semantic,mode=arm,
                hidden=128,heads=4,group_counts=tuple(manifest["group_counts"])
            ).to(device)
            model.load_state_dict(state,strict=True)
            for parameter in model.base.parameters():
                parameter.requires_grad_(False)
            print(f"[m66] START {arm} train={len(train_ids)} "
                  f"held={len(held_ids)}",flush=True)
            train_arm(model,arm,visual,geo,neighborhood,graph,y,names,
                      train_ids,stats,cache,directory,signature,args,device)
            rows=[]
            for task in TRAIN_TASKS:
                ids=np.asarray([int(i) for i in held_ids
                                if names[int(i)]==task],dtype=np.int64)
                result=eval_arm(model,task,ids,visual,geo,neighborhood,
                                graph,y,names,stats,cache,args.batch_size,
                                device)
                result["arm"]=arm
                rows.append(result)
                print(f"[m66] {arm:16s} {task:20s} "
                      f"MSE={result['mse_log1p']:.3f} "
                      f"Pearson={result['pearson']:.3f} "
                      f"bias={result['bias']:+.3f}",flush=True)
            write_csv(complete,rows)
            metrics.extend(rows)
            del model
            if device.type=="cuda":
                torch.cuda.empty_cache()
        write_csv(args.output/"SUMMARY.csv",metrics)
        lines=[
            "# M6.6 source-held-city neighbor × cross-indicator comparison",
            "",
            "Source Pop/Food only. Identical initial weights, visual anchor, "
            "architecture, batch order and 20 epochs; task-neighbor "
            "semantic cross interaction is the controlled difference.",
            "",
            "| Task | Arm | MSE ↓ | Pearson ↑ | Bias |",
            "|---|---|---:|---:|---:|"]
        for row in metrics:
            lines.append(f"| {row['task']} | {row['arm']} | "
                         f"{float(row['mse_log1p']):.3f} | "
                         f"{float(row['pearson']):.3f} | "
                         f"{float(row['bias']):+.3f} |")
        lines.extend(["",
                      "Attention combines K=8 neighboring RemoteCLIP "
                      "images with label-free Population/Food/Education "
                      "definition+relation prompt prototypes, derived "
                      "from source-training imagery only.",
                      "No neighbor B labels, Education supervision or "
                      "official test labels. This is a single-seed "
                      "source-city development experiment, not proof "
                      "of unseen indicator transfer."])
        (args.output/"REPORT.md").write_text("\n".join(lines)+"\n")
        print(f"[m66] COMPLETE {args.output}",flush=True)
    finally:
        store.close()


if __name__=="__main__":
    main()
