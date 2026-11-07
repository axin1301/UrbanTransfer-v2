"""M6.5 full-data neighborhood transfer with source-only city-held validation.

Run --stage dev first: reserve complete cities *within original train
cities*, fit a fresh visual anchor and neighborhood student on other
source train cities; report only held-city Pop/Food metrics. Absolutely
no original held-city UCSI or Education outcomes are used in dev.

Run --stage final separately: use original source-only full visual
anchor and fit M6.5 on all original source train rows, then one-shot
SCUI/UCSI/UCUI development reporting with frozen final-epoch student.
Qwen-VL tokens are read from existing M6.2 hidden.sqlite3, without
encoding. Per-stage atomic epoch checkpoints support resuming.
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
    CachedTokens,validate_full,source_train_targets,do_fit)
from scripts.run_k7_m63_full import (
    PersistentReadOnlyHiddenStore,load_training_tokens,read_reference)
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.k7_m65_neighborhood import load_or_build_neighborhood
from urban_transfer.k7_m65_neighborhood_student import NeighborhoodCityIndicatorResidual


def city_dev_split(frame,original_train,seed=42,fraction=.20):
    """Source-task rows only: reserve whole training cities, no tile leakage."""
    if seed!=42 or not 0<fraction<.5:
        raise ValueError("Fixed source-city dev split required")
    index=np.asarray(original_train,dtype=np.int64)
    tasks=frame.indicator_column.astype(str).to_numpy()
    cities=frame.city.astype(str).to_numpy()
    if not np.isin(tasks[index],TRAIN_TASKS).all():
        raise ValueError("Only original source training rows may enter dev")
    unique=sorted(set(cities[index].tolist()))
    if len(unique)<3:
        raise ValueError("Too few source training cities for city dev")
    rng=np.random.default_rng(seed+65)
    count=max(1,min(len(unique)-2,int(round(len(unique)*fraction))))
    reserved=sorted(rng.permutation(unique)[:count].tolist())
    chosen=np.isin(cities[index],reserved)
    fit=index[~chosen]
    held=index[chosen]
    for name in TRAIN_TASKS:
        if min(int((tasks[ids]==name).sum()) for ids in (fit,held))<2:
            raise ValueError("Insufficient Pop/Food observations in source city holdout")
    if set(cities[fit]) & set(cities[held]):
        raise ValueError("City-level source fit/val leakage")
    return fit,held,reserved


def fit_source_moments(y,names,ids):
    result={}
    for task in TRAIN_TASKS:
        values=np.asarray(
            [float(y[int(i)]) for i in ids if names[int(i)]==task])
        if len(values)<2:
            raise ValueError("Missing training labels for "+task)
        result[task]=(float(values.mean()),max(float(values.std()),.001))
    return result


@torch.inference_mode()
def evaluate_source_or_target(model,split,ids,x,ctx,neighbor,y,names,
                              stats,target_stats,cache,batch_size,device):
    model.eval()
    normalized=[]
    routes=[]
    for first in range(0,len(ids),batch_size):
        ix=ids[first:first+batch_size]
        h=cache.batch(ix,[names[int(i)] for i in ix],device)
        pred,w,*_=model(x[ix].to(device),ctx[ix].to(device),h,
                        neighbor[ix].to(device),return_aux=True)
        normalized.append(pred.cpu().numpy())
        routes.append(w[:,0].cpu().numpy())
    z=np.concatenate(normalized)
    means=np.asarray([
        target_stats[0] if names[int(i)]==HELDOUT_TASK
        else stats[names[int(i)]][0] for i in ids],dtype=np.float64)
    scales=np.asarray([
        target_stats[1] if names[int(i)]==HELDOUT_TASK
        else stats[names[int(i)]][1] for i in ids],dtype=np.float64)
    pred=means+scales*z
    truth=y[torch.as_tensor(ids,dtype=torch.long)].numpy()
    mse=float(np.mean((pred-truth)**2))
    bias=float(pred.mean()-truth.mean())
    pearson=(float(np.corrcoef(pred,truth)[0,1])
             if min(float(pred.std()),float(truth.std()))>1e-10
             else float("nan"))
    return {"model":"m65","partition":split,"n_rows":len(ids),
            "mse_log1p":mse,
            "mae_log1p":float(np.mean(np.abs(pred-truth))),
            "pearson":pearson,
            "relative_mse":mse/max(float(np.mean((means-truth)**2)),1e-12),
            "mean_bias_log1p":bias,
            "pred_std":float(pred.std()),
            "truth_std":float(truth.std()),
            "population_weight":float(np.concatenate(routes).mean())}


def train_neighbor_student(model,x,ctx,neighborhood,y,names,train_ids,stats,
                           cache,args,device,checkpoint,signature):
    target=source_train_targets(train_ids,names,y,stats)
    labels=torch.tensor([TRAIN_TASKS.index(names[int(i)])
                         for i in train_ids],dtype=torch.long)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=1e-4)
    rng=torch.Generator().manual_seed(42)
    first_epoch=0
    if checkpoint.is_file():
        saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if saved.get("signature")!=signature or saved.get("version")!=1:
            raise ValueError("M6.5 checkpoint signature mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["shuffle_rng"])
        torch.set_rng_state(saved["cpu_rng"])
        if device.type=="cuda":
            torch.cuda.set_rng_state(saved["gpu_rng"],device=device)
        first_epoch=int(saved["completed_epoch"])
        if not 1<=first_epoch<=args.epochs:
            raise ValueError("Invalid saved M6.5 epoch")
        print(f"[k7-m65] RESUME completed_epoch={first_epoch}",flush=True)
    for epoch in range(first_epoch,args.epochs):
        start_time=time.monotonic()
        model.train()
        model.base.eval()
        order=torch.randperm(len(train_ids),generator=rng)
        total=0.
        count=0
        for start in range(0,len(order),args.batch_size):
            chosen=order[start:start+args.batch_size]
            ix=train_ids[chosen.numpy()]
            v=x[ix].to(device)
            geo=ctx[ix].to(device)
            nb=neighborhood[ix].to(device)
            labels_batch=labels[chosen].to(device)
            yy=target[chosen].to(device)
            source_h,pop_h,food_h,edu_h=load_training_tokens(
                cache,ix,names,device)
            optimizer.zero_grad(set_to_none=True)
            pred,w,task,fused,baseline,delta=model(
                v,geo,source_h,nb,return_aux=True)
            own=model.source_expert_predictions(
                v,geo,source_h,nb).gather(
                    1,labels_batch[:,None]).squeeze(1)
            separation=model.source_task_separation(
                v,geo,pop_h,food_h,nb)
            align=model.target_image_alignment(v,geo,edu_h,nb)
            loss=(F.mse_loss(pred,yy)+.3*F.mse_loss(own,yy)+
                  .2*separation+.05*align+.02*delta.square().mean())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite M6.5 source loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,2.)
            optimizer.step()
            total+=float(loss.detach().cpu())
            count+=1
            batch=(start//args.batch_size)+1
            if batch%50==0:
                print(f"[k7-m65] epoch={epoch+1}/{args.epochs} "
                      f"batch={batch} elapsed_s={time.monotonic()-start_time:.1f}",
                      flush=True)
        progress={
            "version":1,"signature":signature,"completed_epoch":epoch+1,
            "model":{k:v.detach().cpu().clone()
                     for k,v in model.state_dict().items()},
            "optimizer":optimizer.state_dict(),
            "shuffle_rng":rng.get_state(),"cpu_rng":torch.get_rng_state(),
        }
        if device.type=="cuda":
            progress["gpu_rng"]=torch.cuda.get_rng_state(device=device)
        temp=checkpoint.with_name(checkpoint.name+".tmp")
        torch.save(progress,temp)
        os.replace(temp,checkpoint)
        print(f"[k7-m65] epoch={epoch+1}/{args.epochs} "
              f"source_loss={total/count:.5f} "
              f"elapsed_s={time.monotonic()-start_time:.1f} "
              f"checkpoint_saved=True",flush=True)
    model.eval()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("dev","final"),required=True)
    p.add_argument("--full-dir",type=Path,
                   default=Path("runs/k7/m62_full_seed42"))
    p.add_argument("--reference-dir",type=Path,
                   default=Path("runs/k7/m62_full_models_seed42"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--neighbor-cache",type=Path,
                   default=Path("runs/k7/m65_neighbors_k8.pt"))
    p.add_argument("--k",type=int,default=8)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=p.parse_args()
    if args.k!=8 or args.epochs!=20 or args.batch_size!=256:
        p.error("Fixed M6.5 neighborhood k=8, epochs20, batch256 protocol")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    checkpoint=args.output/"checkpoint_last.pt"
    if args.output.exists() and not checkpoint.is_file():
        raise FileExistsError("Output exists without resumable checkpoint")
    if (args.output/"metrics.csv").is_file():
        raise FileExistsError("M6.5 stage already complete")
    manifest_path=(args.full_dir/"full_manifest.json").resolve()
    manifest=json.loads(manifest_path.read_text())
    frame,x,ctx,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    splits=validate_full(frame,manifest)
    neighborhood=load_or_build_neighborhood(
        frame,x,manifest_path,args.neighbor_cache,k=args.k)
    ds=ImageUrbanDataset(manifest["source_csv"],manifest["image_root"],
                         partition="eval")
    if len(ds.df)!=len(frame) or names!=ds.df.indicator_column.astype(str).tolist():
        raise ValueError("Source tensor and image rows misaligned")
    store=PersistentReadOnlyHiddenStore(args.full_dir/"hidden.sqlite3")
    try:
        cache=CachedTokens(frame,ds,manifest,store,maxsize=512)
        first=cache[int(splits["train"][0]),names[int(splits["train"][0])]]
        device=torch.device(args.device)
        torch.manual_seed(42)
        if device.type=="cuda":
            torch.cuda.manual_seed_all(42)
        if args.stage=="dev":
            train_ids,eval_ids,held_cities=city_dev_split(
                frame,splits["train"])
            stats=fit_source_moments(y,names,train_ids)
            # Training this anchor from SCRATCH is essential: the
            # original full-visual checkpoint has seen held-dev cities.
            from types import SimpleNamespace
            base_args=SimpleNamespace(
                seed=42,epochs=20,batch_size=256,hidden=128,
                heads=4,lr=.001,self_loss=.3)
            print(f"[k7-m65] DEV fresh visual anchor training "
                  f"train={len(train_ids)} held_source={len(eval_ids)} "
                  f"cities={held_cities}",flush=True)
            # do_fit needs text embedding dims even for visual_only;
            # no text cache/model inference required in this branch.
            dummy={t:torch.zeros(1) for t in TRAIN_TASKS}
            visual_anchor=do_fit(
                "visual_only",manifest,x,ctx,y,names,train_ids,
                stats,dummy,cache,base_args,device)
            anchor_state={k:v.detach().cpu()
                          for k,v in visual_anchor.state_dict().items()}
            target_stats=(0.,1.)  # unused: dev labels only Pop/Food
        else:
            train_ids=splits["train"]
            eval_ids=None
            held_cities=[]
            reference,stats,target_stats=read_reference(
                args.reference_dir,manifest_path,manifest)
            anchor_state=reference["model"]
        model=NeighborhoodCityIndicatorResidual(
            x.shape[-1],first.shape[-1],hidden=128,heads=4,
            group_counts=tuple(manifest["group_counts"]),
            max_residual=.5).to(device)
        model.load_visual_baseline(anchor_state)
        for param in model.base.parameters():
            param.requires_grad_(False)
        config={
            "stage":args.stage,"manifest":str(manifest_path),
            "k":args.k,"epochs":args.epochs,"batch_size":args.batch_size,
            "source_train_sha256":hashlib.sha256(
                np.asarray(train_ids,dtype=np.int64).tobytes()).hexdigest(),
            "neighbor_cache":str(args.neighbor_cache.resolve()),
            "held_source_cities":held_cities,
        }
        args.output.mkdir(parents=True,exist_ok=True)
        train_neighbor_student(
            model,x,ctx,neighborhood,y,names,train_ids,stats,cache,
            args,device,checkpoint,config)
        rows=[]
        if args.stage=="dev":
            for task in TRAIN_TASKS:
                ids=np.asarray([int(i) for i in eval_ids
                                if names[int(i)]==task],dtype=np.int64)
                rows.append(evaluate_source_or_target(
                    model,"dev_held_city_"+task,ids,x,ctx,neighborhood,
                    y,names,stats,target_stats,cache,args.batch_size,device))
            # Report no Education metrics, no original UCSI metrics.
        else:
            for partition in ("source_val","SCUI","UCSI","UCUI"):
                rows.append(evaluate_source_or_target(
                    model,partition,splits[partition],x,ctx,neighborhood,
                    y,names,stats,target_stats,cache,args.batch_size,device))
            for task in TRAIN_TASKS:
                ids=np.asarray([int(i) for i in splits["UCSI"]
                                if names[int(i)]==task],dtype=np.int64)
                rows.append(evaluate_source_or_target(
                    model,"UCSI_"+task,ids,x,ctx,neighborhood,
                    y,names,stats,target_stats,cache,args.batch_size,device))
        torch.save({
            "model":{k:v.detach().cpu() for k,v in model.state_dict().items()},
            "config":config,"source_stats":stats,
            "target_stats":target_stats},
            args.output/"m65.pt")
        with (args.output/"metrics.csv").open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (args.output/"config.json").write_text(json.dumps({
            **config,
            "protocol":"Source labels only, no Education fitting. "
            "Development city validation uses freshly source-fit visual anchor. "
            "Final original official test untouched. "
            "Unlabeled source-CSV imagery used for neighborhood lookup."},
            indent=2))
        for row in rows:
            print(f"[k7-m65] {row['partition']:29s} "
                  f"MSE={row['mse_log1p']:.4f} "
                  f"Pearson={row['pearson']:.4f} "
                  f"bias={row['mean_bias_log1p']:+.4f}",flush=True)
        print(f"[k7-m65] {args.stage} COMPLETE {args.output}",flush=True)
    finally:
        store.close()


if __name__=="__main__":
    main()
