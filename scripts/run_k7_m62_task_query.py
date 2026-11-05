"""K7 M6.2: ONE task-queried expert fit, reusing M6.1 frozen MLLM cache.

Zero Education labels in any optimization term. Source train rows only,
source city isolated from UCUI, no final-test files. M6.1 untouched.
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

from scripts.run_k7_m61_student import load_precomputed
from scripts.run_z1_pilot import calibrated_moments,scores
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.k7_mllm_hidden import HiddenStore,TASKS
from urban_transfer.k7_m62_task_query import TaskQueriedRouter
from urban_transfer.ktrans_text import qwen_task_vectors


def validate_split(frame,manifest):
    train=np.asarray(manifest["train"],dtype=np.int64)
    scui=np.asarray(manifest["SCUI"],dtype=np.int64)
    ucui=np.asarray(manifest["UCUI"],dtype=np.int64)
    all_ids=np.concatenate((train,scui,ucui))
    if (not len(train) or not len(scui) or not len(ucui)
        or len(set(all_ids.tolist()))!=len(all_ids)
        or min(all_ids)<0 or max(all_ids)>=len(frame)):
        raise ValueError("Invalid/overlapping M6.1 manifest indices")
    tasks=frame.indicator_column.astype(str).to_numpy()
    unseen=frame.city.astype(str).isin(manifest["held_cities"]).to_numpy()
    if not (np.isin(tasks[train],TRAIN_TASKS).all() and
            not unseen[train].any() and
            np.all(tasks[scui]==HELDOUT_TASK) and not unseen[scui].any() and
            np.all(tasks[ucui]==HELDOUT_TASK) and unseen[ucui].all()):
        raise ValueError("Source-only training and held-city split violated")
    return train,scui,ucui


@torch.inference_mode()
def frozen_metrics(model,visual,context,targets,indices,cache,
                   mean,std,batch_size,device):
    out=[]
    weights=[]
    shuffled=[]
    for start in range(0,len(indices),batch_size):
        ix=indices[start:start+batch_size]
        token=torch.stack([cache[int(i),HELDOUT_TASK] for i in ix]).to(device)
        x=visual[ix].to(device)
        c=context[ix].to(device)
        pred,route,_,_=model(x,c,token,return_aux=True)
        out.append(pred.cpu())
        weights.append(route[:,0].cpu())
        # Keep image/geography unchanged, rotate only MLLM input.
        other=indices[(np.arange(start,start+len(ix))+1)%len(indices)]
        wrong=torch.stack([cache[int(i),HELDOUT_TASK] for i in other]).to(device)
        shuffled.append(model(x,c,wrong).cpu())
    normalized=torch.cat(out).numpy()
    pred=mean+std*normalized
    truth=targets[torch.from_numpy(indices.copy())].numpy()
    metrics=scores(pred,truth)
    baseline=float(np.mean((mean-truth)**2))
    w=torch.cat(weights).numpy()
    mismatch=float(std*(torch.cat(out)-torch.cat(shuffled)).abs().mean())
    return {**metrics,"ratio":metrics["mse_log1p"]/max(baseline,1e-12),
            "population_weight":float(w.mean()),
            "population_weight_std":float(w.std()),
            "food_weight":float(1-w.mean()),
            "mismatch_change_log1p":mismatch}


@torch.inference_mode()
def same_image_prompt_audit(model,x,ctx,train_ids,source_cache,batch_size,device):
    report={}
    predictions={t:[] for t in TASKS}
    weights={t:[] for t in TASKS}
    task_vectors={t:[] for t in TASKS}
    for start in range(0,len(train_ids),batch_size):
        ix=train_ids[start:start+batch_size]
        xv=x[ix].to(device)
        cv=ctx[ix].to(device)
        for t in TASKS:
            z=torch.stack([source_cache[int(i),t] for i in ix]).to(device)
            pred,w,q,_=model(xv,cv,z,return_aux=True)
            predictions[t].append(pred.cpu())
            weights[t].append(w[:,0].cpu())
            task_vectors[t].append(q.cpu())
    p={t:torch.cat(v) for t,v in predictions.items()}
    w={t:torch.cat(v) for t,v in weights.items()}
    q={t:F.normalize(torch.cat(v).float(),dim=-1)
       for t,v in task_vectors.items()}
    for a,b in ((TRAIN_TASKS[0],HELDOUT_TASK),
                (TRAIN_TASKS[1],HELDOUT_TASK),
                (TRAIN_TASKS[0],TRAIN_TASKS[1])):
        key=a+"__vs__"+b
        report[key]={
            "abs_population_weight_delta":float((w[a]-w[b]).abs().mean()),
            "abs_normalized_prediction_delta":float((p[a]-p[b]).abs().mean()),
            "task_query_cosine":float((q[a]*q[b]).sum(-1).mean())}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir",type=Path,
                   default=Path("runs/k7/m61_structured_seed42"))
    p.add_argument("--output",type=Path,
                   default=Path("runs/k7/m62_task_query_seed42"))
    p.add_argument("--text-cache-dir",type=Path,default=Path("runs/ktrans_text"))
    p.add_argument("--text-model",default="Qwen/Qwen3-Embedding-0.6B")
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
    if (a.seed!=42 or a.epochs!=20 or a.batch_size!=128 or a.hidden<8 or
        min(a.self_loss,a.task_loss,a.target_align_loss)<0 or
        a.task_loss==0 or a.target_align_loss==0):
        p.error("Fixed pilot: seed42, epochs20, batch128, positive task and target losses")
    if not torch.cuda.is_available():
        raise RuntimeError("MI250/ROCm device unavailable (torch.cuda)")
    if a.output.resolve()==a.cache_dir.resolve() or a.output.exists():
        raise FileExistsError("Output must be a NEW directory; no overwrites: "+str(a.output))
    manifest=json.loads((a.cache_dir/"manifest.json").read_text())
    frame,x,ctx,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    train_ids,scui,ucui=validate_split(frame,manifest)
    ds=ImageUrbanDataset(manifest["source_csv"],manifest["image_root"],
                         partition="eval")
    if len(ds.df)!=len(frame) or names!=ds.df.indicator_column.astype(str).tolist():
        raise ValueError("Cache/frame row alignment mismatch")
    store=HiddenStore(a.cache_dir/"hidden.sqlite3")
    source_cache=load_precomputed(frame,train_ids,TASKS,ds,store,manifest)
    eval_ids=np.unique(np.concatenate([scui,ucui]))
    eval_cache=load_precomputed(frame,eval_ids,(HELDOUT_TASK,),ds,store,manifest)
    semantics=qwen_task_vectors(root=a.text_cache_dir,model_id=a.text_model,
                                 profile="short",only_cached=True)
    train_frame=frame.iloc[train_ids]
    stats,(target_mean,target_std)=calibrated_moments(
        train_frame,semantics,HELDOUT_TASK,temperature=.2)
    task_names=[names[int(i)] for i in train_ids]
    yy=torch.tensor([(float(y[int(i)])-stats[task][0])/stats[task][1]
                     for i,task in zip(train_ids,task_names)],dtype=torch.float32)
    labels=torch.tensor([TRAIN_TASKS.index(t) for t in task_names],
                        dtype=torch.long)
    source_h=torch.stack([source_cache[int(i),task]
                          for i,task in zip(train_ids,task_names)])
    pop_h=torch.stack([source_cache[int(i),TRAIN_TASKS[0]]
                       for i in train_ids])
    food_h=torch.stack([source_cache[int(i),TRAIN_TASKS[1]]
                        for i in train_ids])
    education_h=torch.stack([source_cache[int(i),HELDOUT_TASK]
                             for i in train_ids])
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device=torch.device("cuda")
    model=TaskQueriedRouter(
        x.shape[1],source_h.shape[-1],hidden=a.hidden,heads=a.heads,
        group_counts=tuple(manifest["group_counts"])).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    src_ids=torch.from_numpy(train_ids.copy())
    visual=x[src_ids].to(device)
    context=ctx[src_ids].to(device)
    source_h=source_h.to(device)
    pop_h=pop_h.to(device)
    food_h=food_h.to(device)
    education_h=education_h.to(device)
    yy=yy.to(device)
    labels=labels.to(device)
    rng=torch.Generator(device="cpu").manual_seed(a.seed)
    print(f"[k7-m62] gpu={torch.cuda.get_device_name(0)} "
          f"train={len(train_ids)} SCUI={len(scui)} UCUI={len(ucui)} "
          f"tokens={source_h.shape[1]} dim={source_h.shape[2]} "
          f"seed={a.seed} trained_models=1",flush=True)
    for epoch in tqdm(range(a.epochs),desc="M6.2 ONE task-queried fit"):
        model.train()
        permutation=torch.randperm(len(train_ids),generator=rng)
        total=0.
        count=0
        for start in range(0,len(permutation),a.batch_size):
            j=permutation[start:start+a.batch_size].to(device)
            opt.zero_grad(set_to_none=True)
            pred=model(visual[j],context[j],source_h[j])
            sup=F.mse_loss(pred,yy[j])
            alone=model.source_expert_predictions(
                visual[j],context[j],source_h[j])
            own=alone.gather(1,labels[j].unsqueeze(1)).squeeze(1)
            self_loss=F.mse_loss(own,yy[j])
            # Same SOURCE image x with BOTH task descriptions. Task
            # classification only sees source task identities, no
            # Education regression labels / pseudo-validation images.
            separation=model.source_task_separation(pop_h[j],food_h[j])
            # Education target definition participates, but only on
            # source images through a weak image-representation alignment.
            align=model.target_image_alignment(education_h[j],visual[j])
            loss=(sup+a.self_loss*self_loss+
                  a.task_loss*separation+a.target_align_loss*align)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step()
            total+=float(loss.detach())
            count+=1
        if epoch in (0,4,9,14,a.epochs-1):
            print(f"[k7-m62] epoch={epoch+1}/{a.epochs} "
                  f"loss={total/count:.4f}",flush=True)
    model.eval()
    result={
        split:frozen_metrics(model,x,ctx,y,ix,eval_cache,
                             target_mean,target_std,a.batch_size,device)
        for split,ix in (("SCUI",scui),("UCUI",ucui))}
    same_image=same_image_prompt_audit(
        model,x,ctx,train_ids,source_cache,a.batch_size,device)
    a.output.mkdir(parents=True,exist_ok=False)
    # M6.2 is an independent checkpoint. Frozen M6.1 cache stays read-only.
    torch.save({
        "model":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "source_stats":stats,"target_stats":[target_mean,target_std],
        "heldout_cities":manifest["held_cities"],
        "group_counts":manifest["group_counts"]},a.output/"model.pt")
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update({
        "source_tasks":list(TRAIN_TASKS),"heldout_task":HELDOUT_TASK,
        "source_train_n":len(train_ids),"scui_n":len(scui),"ucui_n":len(ucui),
        "protocol":"Target Education text on source images only; source labels for regression; Education pseudo-dev labels for reporting only. No official final test.",
        "cache_manifest_path":str((a.cache_dir/"manifest.json").resolve())})
    (a.output/"config.json").write_text(json.dumps(config,indent=2))
    with (a.output/"metrics.csv").open("w",newline="") as fd:
        rows=[{"partition":k,**v} for k,v in result.items()]
        writer=csv.DictWriter(fd,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.output/"same_image_prompt_audit.json").write_text(
        json.dumps(same_image,indent=2))
    for split,metrics in result.items():
        print(f"[k7-m62] {split} MSE={metrics['mse_log1p']:.3f} "
              f"Pearson={metrics['pearson']:.3f} "
              f"Ratio={metrics['ratio']:.3f} "
              f"PopWt={metrics['population_weight']:.3f} "
              f"PopWtStd={metrics['population_weight_std']:.3f} "
              f"MismatchChange={metrics['mismatch_change_log1p']:.3f}",flush=True)
    for pair,d in same_image.items():
        print(f"[k7-m62] {pair} task_query_cos={d['task_query_cosine']:.4f} "
              f"router_abs_delta={d['abs_population_weight_delta']:.4f} "
              f"pred_abs_delta={d['abs_normalized_prediction_delta']:.4f}",flush=True)
    print("[k7-m62] COMPLETE "+str(a.output),flush=True)


if __name__=="__main__":
    main()
