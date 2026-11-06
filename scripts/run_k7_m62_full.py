"""Full-source M6.2 study: two controls + task-query student.

Frozen RemoteCLIP tensor cache and resumable frozen Qwen2.5-VL
64×2048 hidden cache; minibatch SQLite lookup (no full hidden tensor
in GPU RAM). The ONLY fit labels are Pop/Food training rows.

Holdout policy:
  - city heldout / indicator heldout defined in full_manifest.json
  - SOURCE-only source_val is for reporting; NO dev/checkpoint tuning.
  - SCUI/UCUI Education labels are NEVER accessed until after ALL
    three fixed-20-epoch fits have finished.
  - official final test files are NEVER opened.
"""
import argparse
import csv
import json
import sys
from functools import lru_cache
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.run_k7_m62_fixed_comparison import (
    TwoExpertControl,task_tensor,eval_split)
from scripts.run_z1_pilot import calibrated_moments
from scripts.prepare_k7_m62_full import build_job_index
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.k7_mllm_hidden import HiddenStore
from urban_transfer.k7_m61_structured import structured_key
from urban_transfer.k7_m62_task_query import TaskQueriedRouter
from urban_transfer.ktrans_text import qwen_task_vectors


def validate_full(frame,manifest):
    if int(manifest["seed"])!=42:
        raise ValueError("Full source pilot seed mismatch")
    keys=("train","source_val","SCUI","UCSI","UCUI")
    splits={k:np.asarray(manifest[k],dtype=np.int64) for k in keys}
    indices=np.concatenate(list(splits.values()))
    if (len(indices)!=len(set(indices.tolist())) or min(indices)<0 or
        max(indices)>=len(frame) or any(len(a)<2 for a in splits.values())):
        raise ValueError("Invalid or overlapping full-data manifest")
    tasks=frame.indicator_column.astype(str).to_numpy()
    held=frame.city.astype(str).isin(manifest["held_cities"]).to_numpy()
    for name in ("train","source_val"):
        idx=splits[name]
        if held[idx].any() or not np.isin(tasks[idx],TRAIN_TASKS).all():
            raise ValueError(name+" contains unseen city or held indicator")
    if not (np.isin(tasks[splits["UCSI"]],TRAIN_TASKS).all()
        and held[splits["UCSI"]].all()
        and (tasks[splits["SCUI"]]==HELDOUT_TASK).all()
        and not held[splits["SCUI"]].any()
        and (tasks[splits["UCUI"]]==HELDOUT_TASK).all()
        and held[splits["UCUI"]].all()):
        raise ValueError("Full-data task×city evaluation partition leakage")
    for source in TRAIN_TASKS:
        for part in ("train","source_val","UCSI"):
            if int((tasks[splits[part]]==source).sum())<2:
                raise ValueError("Too few "+source+" rows in "+part)
    return splits


class CachedTokens:
    """CPU-backed on-demand read of frozen SQLite Qwen token arrays.

    Key resolution occurs once per needed row/task; reads use small LRU
    to avoid keeping all 64×2048 tensors in RAM. No MLLM inference.
    """
    def __init__(self,frame,ds,manifest,store,maxsize=128):
        self.frame=frame
        self.ds=ds
        self.manifest=manifest
        self.store=store
        self.names=frame.indicator_column.astype(str).tolist()
        self._lookup=lru_cache(maxsize=maxsize)(self._load)

    def _load(self,idx,task):
        row=self.ds.df.iloc[int(idx)]
        path=str(self.ds.find_image(row))
        key=structured_key(
            path,task,self.manifest["model_id"],
            self.manifest["max_pixels"],self.manifest["group_counts"])
        token=self.store.get(key)
        if token is None:
            raise FileNotFoundError(
                f"Qwen-VL hidden missing for {idx}/{task}; run full cache first")
        if token.shape[0]!=sum(self.manifest["group_counts"]):
            raise ValueError("Wrong cached token length")
        return token

    def __getitem__(self,key):
        idx,task=key
        return self._lookup(int(idx),task)

    def batch(self,indices,tasks,device):
        return torch.stack([self[int(i),str(task)]
                            for i,task in zip(indices,tasks)]).to(device)


def source_train_targets(train_ids,names,y,stats):
    return torch.tensor([
        (float(y[int(i)])-stats[names[int(i)]][0])/
        stats[names[int(i)]][1] for i in train_ids],dtype=torch.float32)


@torch.inference_mode()
def evaluate(model_name,model,split,indices,x,ctx,y,names,stats,
             unseen_stats,text_embeddings,token_cache,batch_size,device):
    predicted=[]
    route=[]
    for first in range(0,len(indices),batch_size):
        ix=indices[first:first+batch_size]
        visual=x[ix].to(device)
        geo=ctx[ix].to(device)
        if model_name=="m62":
            token=token_cache.batch(ix,[names[int(i)] for i in ix],device)
            z,w,_,_=model(visual,geo,token,return_aux=True)
        else:
            txt=task_tensor(names,text_embeddings,ix,device) if (
                model_name=="text_router") else None
            z,w=model(visual,geo,txt,return_aux=True)
        predicted.append(z.detach().cpu().numpy())
        route.append(w[:,0].detach().cpu().numpy())
    standardized=np.concatenate(predicted)
    m=np.asarray([unseen_stats[0] if names[int(i)]==HELDOUT_TASK
                  else stats[names[int(i)]][0] for i in indices])
    s=np.asarray([unseen_stats[1] if names[int(i)]==HELDOUT_TASK
                  else stats[names[int(i)]][1] for i in indices])
    truth=y[torch.as_tensor(indices,dtype=torch.long)].numpy()
    z=m+s*standardized
    mse=float(np.mean((z-truth)**2))
    pearson=(float(np.corrcoef(z,truth)[0,1])
             if min(float(z.std()),float(truth.std()))>1e-10 else float("nan"))
    pop=np.concatenate(route)
    base=float(np.mean((m-truth)**2))
    return {"model":model_name,"partition":split,
            "n_rows":int(len(indices)),"mse_log1p":mse,
            "mae_log1p":float(np.mean(np.abs(z-truth))),
            "pearson":pearson,
            "source_mean_mse":base,
            "relative_mse":mse/max(base,1e-12),
            "population_weight":float(pop.mean()),
            "population_weight_std":float(pop.std()),
            "prediction_mean":float(z.mean()),"truth_mean":float(truth.mean())}


def do_fit(variant,manifest,x,ctx,y,names,train_ids,stats,
           embeddings,tokens,args,device):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    embed_dim=embeddings[TRAIN_TASKS[0]].numel()
    if variant=="m62":
        first=tokens[int(train_ids[0]),names[int(train_ids[0])]]
        model=TaskQueriedRouter(
            x.shape[-1],first.shape[-1],hidden=args.hidden,
            heads=args.heads,group_counts=tuple(manifest["group_counts"]))
    else:
        model=TwoExpertControl(
            x.shape[-1],text_dim=embed_dim if variant=="text_router" else 0,
            hidden=args.hidden)
    model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    label=source_train_targets(train_ids,names,y,stats)
    label_idx=torch.tensor([TRAIN_TASKS.index(names[int(i)]) for i in train_ids],
                           dtype=torch.long)
    rng=torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        permutation=torch.randperm(len(train_ids),generator=rng)
        losses=[]
        for first in range(0,len(train_ids),args.batch_size):
            local=permutation[first:first+args.batch_size]
            ix=train_ids[local.numpy()]
            visual=x[ix].to(device)
            geo=ctx[ix].to(device)
            target=label[local].to(device)
            task_indices=label_idx[local].to(device)
            opt.zero_grad(set_to_none=True)
            if variant=="m62":
                h=tokens.batch(ix,[names[int(i)] for i in ix],device)
                pop=tokens.batch(ix,[TRAIN_TASKS[0]]*len(ix),device)
                food=tokens.batch(ix,[TRAIN_TASKS[1]]*len(ix),device)
                edu=tokens.batch(ix,[HELDOUT_TASK]*len(ix),device)
                pred=model(visual,geo,h)
                direct=model.source_expert_predictions(visual,geo,h)
                own=direct.gather(1,task_indices[:,None]).squeeze(1)
                classification=model.source_task_separation(pop,food)
                alignment=model.target_image_alignment(edu,visual)
                loss=(F.mse_loss(pred,target)+args.self_loss*
                      F.mse_loss(own,target)+args.task_loss*classification+
                      args.target_align_loss*alignment)
            else:
                text=task_tensor(names,embeddings,ix,device) if (
                    variant=="text_router") else None
                pred=model(visual,geo,text)
                direct=model.source_expert_predictions(visual,geo,text)
                own=direct.gather(1,task_indices[:,None]).squeeze(1)
                loss=(F.mse_loss(pred,target)+
                      args.self_loss*F.mse_loss(own,target))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step()
            losses.append(float(loss.detach()))
        if (epoch==0 or (epoch+1)%5==0 or epoch+1==args.epochs):
            print(f"[k7-full-fit] {variant} epoch={epoch+1}/{args.epochs} "
                  f"source_train_loss={np.mean(losses):.5f}",flush=True)
    return model.eval()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,
                   default=Path("runs/k7/m62_full_seed42"))
    p.add_argument("--output",type=Path,
                   default=Path("runs/k7/m62_full_models_seed42"))
    p.add_argument("--text-cache-dir",type=Path,default=Path("runs/ktrans_text"))
    p.add_argument("--text-model",default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--hidden",type=int,default=128)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--self-loss",type=float,default=.3)
    p.add_argument("--task-loss",type=float,default=.2)
    p.add_argument("--target-align-loss",type=float,default=.05)
    a=p.parse_args()
    if (a.seed!=42 or a.epochs!=20 or a.batch_size!=128
        or a.hidden!=128 or a.heads!=4 or a.lr!=.001
        or a.self_loss!=.3 or a.task_loss!=.2
        or a.target_align_loss!=.05):
        p.error("Predeclared fixed 20-epoch seed42 M6.2 protocol required")
    if a.output.exists() or a.output.resolve()==a.run_dir.resolve():
        raise FileExistsError("Independent output directory must not exist")
    if a.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("MI250/ROCm GPU unavailable")
    manifest=json.loads((a.run_dir/"full_manifest.json").read_text())
    frame,x,ctx,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    ids=validate_full(frame,manifest)
    ds=ImageUrbanDataset(manifest["source_csv"],manifest["image_root"],
                         partition="eval")
    if len(ds.df)!=len(frame) or ds.df.indicator_column.astype(str).tolist()!=names:
        raise ValueError("Image/source tensor cache row order mismatch")
    store=HiddenStore(a.run_dir/"hidden.sqlite3")
    tokens=CachedTokens(frame,ds,manifest,store)
    embeddings=qwen_task_vectors(
        root=a.text_cache_dir,model_id=a.text_model,
        profile="short",only_cached=True)
    source_frame=frame.iloc[ids["train"]]
    stats,unseen_stats=calibrated_moments(
        source_frame,embeddings,HELDOUT_TASK,temperature=.2)
    for task in TRAIN_TASKS:
        if task not in stats:
            raise ValueError("Missing source training task statistics: "+task)
    # All training is completed before SCUI/UCSI/UCUI labels are accessed.
    device=torch.device(a.device)
    models={}
    for variant in ("visual_only","text_router","m62"):
        print(f"[k7-full-fit] START {variant} source={len(ids['train'])} "
              f"val={len(ids['source_val'])}",flush=True)
        models[variant]=do_fit(
            variant,manifest,x,ctx,y,names,ids["train"],stats,
            embeddings,tokens,a,device)
    # Independent checkpoints; evaluate all in final training epoch
    # state, not the best Education pseudo-dev checkpoint.
    results=[]
    for variant,model in models.items():
        for split in ("source_val","SCUI","UCSI","UCUI"):
            ix=ids[split]
            results.append(evaluate(
                variant,model,split,ix,x,ctx,y,names,stats,
                unseen_stats,embeddings,tokens,a.batch_size,device))
        for task in TRAIN_TASKS:
            ix=np.asarray([int(i) for i in ids["UCSI"]
                           if names[int(i)]==task],dtype=np.int64)
            results.append(evaluate(
                variant,model,"UCSI_"+task,ix,x,ctx,y,names,stats,
                unseen_stats,embeddings,tokens,a.batch_size,device))
    a.output.mkdir(parents=True,exist_ok=False)
    for variant,model in models.items():
        torch.save({"model":{k:v.cpu().detach() for k,v in model.state_dict().items()},
                    "variant":variant,"seed":42,"epochs":20,
                    "heldout_cities":manifest["held_cities"],
                    "source_stats":stats,"unseen_stats":unseen_stats,
                    "manifest_path":str((a.run_dir/"full_manifest.json").resolve())},
                   a.output/(variant+".pt"))
    with (a.output/"metrics.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (a.output/"config.json").write_text(json.dumps({
        **{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        "protocol":"FULL source train.csv. Pop/Food supervised only. Education labels accessed post-fit for descriptive SCUI/UCUI scoring. Official test files untouched.",
        "n_source":len(ids["train"]),"n_source_val":len(ids["source_val"]),
        "n_scui":len(ids["SCUI"]),"n_ucsi":len(ids["UCSI"]),
        "n_ucui":len(ids["UCUI"])},indent=2))
    for row in results:
        print(f"[k7-full] {row['model']:12s} {row['partition']:28s} "
              f"n={row['n_rows']} MSE={row['mse_log1p']:.4f} "
              f"Pearson={row['pearson']:.4f} "
              f"Ratio={row['relative_mse']:.4f}",flush=True)
    print(f"[k7-full] COMPLETE {a.output}",flush=True)


if __name__=="__main__":
    main()
