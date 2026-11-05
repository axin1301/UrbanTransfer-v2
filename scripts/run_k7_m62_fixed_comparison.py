"""Fixed, source-only matched comparison: 2 new baselines + FROZEN M6.2.

Source supervised train=128 from M6.1 manifest; UCSI=64 held-city
Population/Food, SCUI/UCUI 64 Education each. All test rows originate
solely from city_indicator_splits/train.csv. No official test touched.

The baselines train for the predeclared 20 epochs with the SAME source
regression/main+matching-expert auxiliary objectives. M6.2 is NOT
retrained; its preexisting checkpoint supplies the direct comparison.
The baselines are architecture controls, not parameter/loss-matched
causal ablations of M6.2's extra alignment/classification losses.
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
from torch import nn

from scripts.run_k7_m61_student import load_precomputed
from scripts.run_k7_m62_task_query import validate_split
from scripts.run_z1_pilot import scores
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.k7_mllm_hidden import HiddenStore
from urban_transfer.k7_m61_structured import structured_key
from urban_transfer.k7_m62_task_query import TaskQueriedRouter
from urban_transfer.ktrans_text import qwen_task_vectors


class TwoExpertControl(nn.Module):
    """Same two expert feature mixture; no MLLM hidden states."""

    def __init__(self,visual_dim,text_dim=0,hidden=128):
        super().__init__()
        self.text_dim=int(text_dim)
        self.visual=nn.Sequential(
            nn.LayerNorm(visual_dim+4),nn.Linear(visual_dim+4,hidden),
            nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        if self.text_dim:
            self.text=nn.Sequential(nn.LayerNorm(text_dim),
                                    nn.Linear(text_dim,hidden),nn.GELU())
        self.condition=nn.Sequential(
            nn.LayerNorm(hidden*(2 if self.text_dim else 1)),
            nn.Linear(hidden*(2 if self.text_dim else 1),hidden),
            nn.GELU())
        self.gate=nn.Sequential(nn.Linear(hidden,hidden),nn.Sigmoid())
        self.router=nn.Sequential(nn.Linear(hidden,2))
        self.experts=nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),
                          nn.GELU(),nn.Linear(hidden,hidden))
            for _ in range(2)])
        self.out=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,1))

    def forward(self,visual,context,text=None,mask=None,return_aux=False):
        h=self.visual(torch.cat([visual.float(),context.float()],dim=-1))
        if self.text_dim:
            if text is None or text.shape!=(len(h),self.text_dim):
                raise ValueError("Text-conditioned baseline needs task embedding")
            cond=self.condition(torch.cat([h,self.text(text.float())],dim=-1))
        else:
            if text is not None:
                raise ValueError("Visual-only baseline must not receive text")
            cond=self.condition(h)
        experts=torch.stack([part(h) for part in self.experts],dim=1)
        experts=experts*self.gate(cond).unsqueeze(1)
        logits=self.router(cond)
        if mask is not None:
            if mask.shape!=logits.shape or not bool(mask.any(-1).all()):
                raise ValueError("At least one expert route is required")
            logits=logits.masked_fill(~mask,-1e9)
        weights=F.softmax(logits,dim=-1)
        pred=self.out((experts*weights.unsqueeze(-1)).sum(1)).squeeze(-1)
        return (pred,weights) if return_aux else pred

    def source_expert_predictions(self,x,ctx,text=None):
        output=[]
        for i in range(2):
            mask=torch.zeros(len(x),2,dtype=torch.bool,device=x.device)
            mask[:,i]=True
            output.append(self(x,ctx,text,mask))
        return torch.stack(output,dim=1)


def assert_protocol(frame,manifest,ucsi_manifest):
    train,scui,ucui=validate_split(frame,manifest)
    ucsi=np.asarray(ucsi_manifest["UCSI"],dtype=np.int64)
    if (len(ucsi)!=64 or len(set(ucsi.tolist()))!=64
        or min(ucsi)<0 or max(ucsi)>=len(frame)
        or set(ucsi.tolist())&set(np.concatenate([train,scui,ucui]).tolist())
        or list(ucsi_manifest["held_cities"])!=list(manifest["held_cities"])
        or tuple(ucsi_manifest["source_tasks"])!=TRAIN_TASKS):
        raise ValueError("UCSI sampling/manifest leakage or mismatch")
    subset=frame.iloc[ucsi]
    if (not subset.city.astype(str).isin(manifest["held_cities"]).all()
        or not subset.indicator_column.astype(str).isin(TRAIN_TASKS).all()
        or set(subset.indicator_column.astype(str))!=set(TRAIN_TASKS)):
        raise ValueError("UCSI MUST comprise held-city source indicators")
    if len(train)!=128 or len(scui)!=64 or len(ucui)!=64:
        raise ValueError("Pilot sample sizes do not match fixed protocol")
    if set(frame.iloc[train].indicator_column)!=set(TRAIN_TASKS):
        raise ValueError("Both source experts need source train labels")
    return {"train":train,"SCUI":scui,"UCSI":ucsi,"UCUI":ucui}


def load_ucsi_hidden(frame,ids,ds,base,run_dir):
    store=HiddenStore(run_dir/"ucsi_hidden.sqlite3")
    cache={}
    for ix in ids:
        i=int(ix)
        task=str(frame.iloc[i].indicator_column)
        path=str(ds.find_image(ds.df.iloc[i]))
        key=structured_key(
            path,task,base["model_id"],base["max_pixels"],base["group_counts"])
        value=store.get(key)
        if value is None:
            raise FileNotFoundError(
                f"UCSI hidden missing for row {i}; run cache_k7_m62_ucsi.py first")
        if value.shape[0]!=sum(base["group_counts"]):
            raise ValueError("UCSI token count mismatch")
        cache[(i,task)]=value
    return cache


def task_tensor(names,embeddings,indices,device):
    return torch.stack([
        embeddings[str(names[int(i)])].float().flatten()
        for i in indices]).to(device)


def eval_split(model_name,model,split,ids,x,ctx,y,names,stats,target_stats,
               embeddings,device,hidden_cache=None,batch_size=64):
    mean=np.asarray([
        target_stats[0] if names[int(i)]==HELDOUT_TASK
        else stats[names[int(i)]][0] for i in ids],dtype=np.float64)
    scale=np.asarray([
        target_stats[1] if names[int(i)]==HELDOUT_TASK
        else stats[names[int(i)]][1] for i in ids],dtype=np.float64)
    output=[]
    weights=[]
    model.eval()
    with torch.inference_mode():
        for first in range(0,len(ids),batch_size):
            j=ids[first:first+batch_size]
            v=x[j].to(device)
            c=ctx[j].to(device)
            if model_name=="m62":
                tokens=torch.stack([
                    hidden_cache[(int(i),names[int(i)])] for i in j]).to(device)
                pred,w,_,_=model(v,c,tokens,return_aux=True)
            else:
                task=(task_tensor(names,embeddings,j,device)
                      if model_name=="text_router" else None)
                pred,w=model(v,c,task,return_aux=True)
            output.append(pred.cpu().numpy())
            weights.append(w[:,0].cpu().numpy())
    prediction=mean+scale*np.concatenate(output)
    truth=y[torch.as_tensor(ids,dtype=torch.long)].numpy()
    metric=scores(prediction,truth)
    baseline=float(np.mean((mean-truth)**2))
    route=np.concatenate(weights)
    return {"model":model_name,"partition":split,
            **metric,"source_mean_mse":baseline,
            "relative_mse":metric["mse_log1p"]/max(baseline,1e-12),
            "population_weight":float(route.mean()),
            "population_weight_std":float(route.std()),
            "mean_prediction":float(prediction.mean()),
            "mean_truth":float(truth.mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir",type=Path,
                   default=Path("runs/k7/m61_structured_seed42"))
    p.add_argument("--m62-dir",type=Path,
                   default=Path("runs/k7/m62_task_query_seed42"))
    p.add_argument("--comparison-dir",type=Path,
                   default=Path("runs/k7/m62_fixed_comparison_seed42"))
    p.add_argument("--text-cache-dir",type=Path,default=Path("runs/ktrans_text"))
    p.add_argument("--text-model",default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--output",type=Path,
                   default=Path("runs/k7/m62_fixed_baselines_seed42"))
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--hidden",type=int,default=128)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--self-loss",type=float,default=.3)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    a=p.parse_args()
    if (a.seed!=42 or a.epochs!=20 or a.batch_size!=128
        or a.hidden!=128 or a.lr!=.001 or a.self_loss!=.3):
        p.error("Pre-registered baseline settings: seed42 epochs20 batch128 hidden128 lr.001 self.3")
    for private in (a.cache_dir,a.m62_dir,a.comparison_dir):
        if a.output.resolve()==private.resolve():
            raise ValueError("Baseline output must not overwrite any existing runs")
    if a.output.exists():
        raise FileExistsError("Baseline output must be new: "+str(a.output))
    if a.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA GPU unavailable")
    device=torch.device(a.device)
    base=json.loads((a.cache_dir/"manifest.json").read_text())
    add=json.loads((a.comparison_dir/"ucsi_manifest.json").read_text())
    config=json.loads((a.m62_dir/"config.json").read_text())
    checkpoint=torch.load(a.m62_dir/"model.pt",map_location="cpu",
                          weights_only=True)
    if (tuple(config["source_tasks"])!=TRAIN_TASKS or
        int(config["seed"])!=42 or
        Path(config["cache_dir"]).resolve()!=a.cache_dir.resolve() or
        Path(add["base_manifest"]).resolve()!=(a.cache_dir/"manifest.json").resolve() or
        list(checkpoint["heldout_cities"])!=list(base["held_cities"]) or
        list(checkpoint["group_counts"])!=list(base["group_counts"])):
        raise ValueError("Source/M6.2/UCSI split or checkpoint mismatch")
    frame,x,ctx,y,names=read_tensor_cache(
        Path(base["tensor_cache"]),Path(base["source_csv"]),
        Path(base["image_root"]),Path(base["feature_dir"]),
        base["remoteclip_model"])
    ids=assert_protocol(frame,base,add)
    ds=ImageUrbanDataset(base["source_csv"],base["image_root"],partition="eval")
    if len(ds.df)!=len(frame) or ds.df.indicator_column.astype(str).tolist()!=names:
        raise ValueError("Source image/RemoteCLIP ordering mismatch")
    m61=HiddenStore(a.cache_dir/"hidden.sqlite3")
    source_cache=load_precomputed(
        frame,ids["train"],TRAIN_TASKS,ds,m61,base)
    education_cache=load_precomputed(
        frame,np.concatenate((ids["SCUI"],ids["UCUI"])),
        (HELDOUT_TASK,),ds,m61,base)
    ucsi_cache=load_ucsi_hidden(frame,ids["UCSI"],ds,base,a.comparison_dir)
    hidden_cache={**source_cache,**education_cache,**ucsi_cache}
    embeddings=qwen_task_vectors(
        root=a.text_cache_dir,model_id=a.text_model,
        profile="short",only_cached=True)
    if not set((*TRAIN_TASKS,HELDOUT_TASK)).issubset(embeddings):
        raise ValueError("Missing source/target frozen Qwen text embeddings")
    stats={k:(float(v[0]),float(v[1]))
           for k,v in checkpoint["source_stats"].items()}
    target_stats=tuple(map(float,checkpoint["target_stats"]))
    for name in TRAIN_TASKS:
        if not stats[name][1]>0:
            raise ValueError("Invalid source standard deviation")
    m62=TaskQueriedRouter(
        x.shape[-1],
        hidden_cache[(int(ids["train"][0]),names[int(ids["train"][0])])].shape[-1],
        hidden=int(config["hidden"]),heads=int(config["heads"]),
        group_counts=tuple(base["group_counts"]))
    m62.load_state_dict(checkpoint["model"],strict=True)
    m62=m62.to(device).eval()
    embedding_dim=embeddings[TRAIN_TASKS[0]].numel()
    train_ids=ids["train"]
    train_names=[names[int(i)] for i in train_ids]
    targets=torch.tensor([
        (float(y[int(i)])-stats[name][0])/stats[name][1]
        for i,name in zip(train_ids,train_names)],dtype=torch.float32).to(device)
    label_index=torch.tensor([
        TRAIN_TASKS.index(name) for name in train_names],
        device=device,dtype=torch.long)
    source_x=x[train_ids].to(device)
    source_ctx=ctx[train_ids].to(device)
    source_text=task_tensor(names,embeddings,train_ids,device)
    result=[]
    trained={}
    for variant in ("visual_only","text_router"):
        torch.manual_seed(a.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(a.seed)
        model=TwoExpertControl(
            x.shape[-1],
            text_dim=0 if variant=="visual_only" else embedding_dim,
            hidden=a.hidden).to(device)
        optim=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
        generator=torch.Generator().manual_seed(a.seed)
        for epoch in range(a.epochs):
            model.train()
            order=torch.randperm(len(train_ids),generator=generator)
            total=0.
            for first in range(0,len(order),a.batch_size):
                j=order[first:first+a.batch_size].to(device)
                text=source_text[j] if variant=="text_router" else None
                optim.zero_grad(set_to_none=True)
                pred=model(source_x[j],source_ctx[j],text)
                own=model.source_expert_predictions(
                    source_x[j],source_ctx[j],text).gather(
                        1,label_index[j].unsqueeze(1)).squeeze(1)
                loss=(F.mse_loss(pred,targets[j])+
                      a.self_loss*F.mse_loss(own,targets[j]))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
                optim.step()
                total+=float(loss.detach())
            if epoch in (0,4,9,14,19):
                print(f"[k7-fixed] {variant} epoch={epoch+1}/20 "
                      f"loss={total:.4f}",flush=True)
        model.eval()
        trained[variant]={k:v.detach().cpu().clone()
                          for k,v in model.state_dict().items()}
        for split in ("SCUI","UCSI","UCUI"):
            row=eval_split(variant,model,split,ids[split],x,ctx,y,names,
                           stats,target_stats,embeddings,device)
            result.append(row)
    # Original M6.2 checkpoint is evaluated unchanged (not retrained).
    for split in ("SCUI","UCSI","UCUI"):
        row=eval_split("m62",m62,split,ids[split],x,ctx,y,names,
                       stats,target_stats,embeddings,device,
                       hidden_cache=hidden_cache)
        result.append(row)
    # Match existing preprinted M6.2 figures within rounding when
    # evaluating SCUI/UCUI with this exact checkpoint.
    for split in ("SCUI","UCUI"):
        row=next(r for r in result
                 if r["model"]=="m62" and r["partition"]==split)
        if not np.isfinite(row["mse_log1p"]):
            raise ValueError("Frozen M6.2 evaluation is nonfinite")
    a.output.mkdir(parents=True,exist_ok=False)
    for variant,weights in trained.items():
        torch.save({"model":weights,"variant":variant,
                    "source_tasks":list(TRAIN_TASKS),
                    "heldout_cities":base["held_cities"],
                    "training_indices":train_ids.tolist(),
                    "seed":a.seed,"epochs":a.epochs},
                   a.output/(variant+".pt"))
    with (a.output/"metrics.csv").open("w",newline="") as fd:
        writer=csv.DictWriter(fd,fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    (a.output/"config.json").write_text(json.dumps({
        **{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        "partition_indices":{k:v.tolist() for k,v in ids.items()},
        "source_tasks":list(TRAIN_TASKS),"target_task":HELDOUT_TASK,
        "protocol":"Predeclared 128 source train / 64 SCUI / 64 UCSI / 64 UCUI. Target labels only scored. M6.2 frozen checkpoint. Baseline architecture controls have no M6.2 auxiliary task separation or target alignment."},indent=2))
    for row in result:
        print(f"[k7-fixed] {row['model']:12s} {row['partition']:4s} "
              f"n={row['n_rows']} MSE={row['mse_log1p']:.3f} "
              f"Pearson={row['pearson']:.3f} Ratio={row['relative_mse']:.3f}",
              flush=True)
    print("[k7-fixed] COMPLETE "+str(a.output),flush=True)


if __name__=="__main__":
    main()
