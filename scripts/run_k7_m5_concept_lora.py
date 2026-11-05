"""K7 M5: one concept-grounded target-text LoRA distillation training, Education DEV evaluation.

During training, align LoRA ranks to frozen RemoteCLIP concept scores,
and distill Qwen target-relation-to-concept distributions on source visuals.
After training, use the SAME checkpoint to compare correct/swapped/generic
directed relation texts; no extra fits or heldout-task label tuning.
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

from scripts.run_k7_m1 import get_cards
from scripts.run_k7_m2 import build_tokens
from scripts.audit_k7_m2_text import generic_embeddings,relation_conditions
from scripts.run_z1_pilot import calibrated_moments,scores
from urban_transfer.data_m1 import stratified_sample
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.ktrans_text import qwen_task_vectors
from urban_transfer.k7_concept_lora import (ConceptLoRAComposer, rank_targets,
    concept_supervision, rank_distillation)
from urban_transfer.k7_m5_concepts import get_concept_anchors,visual_concept_targets


def evaluate(model,x,ctx,indices,sem,source_sem,edges,flags,mean,std,
             truth,batch_size,device):
    outputs,mixes,gates=[],[],[]
    with torch.inference_mode():
        for first in range(0,len(indices),batch_size):
            ix=torch.as_tensor(indices[first:first+batch_size],dtype=torch.long)
            b=len(ix)
            pred,weight,rank_gate=model(x[ix].to(device),ctx[ix].to(device),
                sem.expand(b,-1),source_sem,edges.expand(b,-1,-1),
                flags.expand(b,-1),return_aux=True)
            outputs.append(pred.cpu())
            mixes.append(weight.cpu())
            gates.append(rank_gate.cpu())
    estimate=mean+std*torch.cat(outputs).numpy()
    result=scores(estimate,truth)
    baseline=float(np.mean((mean-truth)**2))
    weights=torch.cat(mixes).mean(0).numpy()
    ranks=torch.cat(gates).mean((0,2)).numpy()
    return estimate,{
        "mse_log1p":result["mse_log1p"],
        "pearson":result["pearson"],
        "ratio":result["mse_log1p"]/max(baseline,1e-12),
        "population_attention":float(weights[0]),
        "food_attention":float(weights[1]),
        "population_rank_gate":float(ranks[0]),
        "food_rank_gate":float(ranks[1]),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv",type=Path,required=True)
    p.add_argument("--image-root",type=Path,required=True)
    p.add_argument("--feature-dir",type=Path,default=Path("runs/m3_features"))
    p.add_argument("--tensor-cache",type=Path,
                   default=Path("runs/k7/e1_e2_v1/_source_tensors.pt"))
    p.add_argument("--remoteclip-model",default="ViT-B-32")
    p.add_argument("--text-cache-dir",type=Path,default=Path("runs/ktrans_text"))
    p.add_argument("--text-model",default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--output",type=Path,default=Path("runs/k7/m5_concept_lora_seed42"))
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--train-limit",type=int,default=20000)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--hidden",type=int,default=128)
    p.add_argument("--rank",type=int,default=8)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--self-dropout",type=float,default=.5)
    p.add_argument("--self-loss",type=float,default=.3)
    p.add_argument("--city-fraction",type=float,default=.25)
    p.add_argument("--scale-temperature",type=float,default=.2)
    p.add_argument("--kd-weight",type=float,default=0.5)
    p.add_argument("--concept-weight",type=float,default=0.25)
    p.add_argument("--teacher-temperature",type=float,default=0.15)
    a=p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("K7 M5 requires MI250/ROCm GPU allocation")
    if a.kd_weight<=0 or a.concept_weight<=0 or a.teacher_temperature<=0:
        p.error("Positive KD weight and teacher temperature required")
    if a.seed!=42:
        raise ValueError("M5 concept-aware pilot is fixed to seed42")
    if (a.epochs<1 or a.batch_size<1 or a.rank<1 or a.rank%4 or
        not 0<=a.self_dropout<1 or a.self_loss<0 or
        not 0<a.city_fraction<.5):
        p.error("Invalid M5 budget")
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError("Refusing to overwrite M5 results: "+str(a.output))

    frame,x,ctx,y,names=read_tensor_cache(
        a.tensor_cache,a.source_csv,a.image_root,a.feature_dir,a.remoteclip_model)
    embeddings=qwen_task_vectors(root=a.text_cache_dir,model_id=a.text_model,
                                profile="short",only_cached=True)
    cards=get_cards(a.text_cache_dir,a.text_model)
    cities=sorted(frame.city.astype(str).unique())
    if len(cities)<3:
        raise ValueError("Three or more source cities required")
    n=max(1,min(len(cities)-2,round(len(cities)*a.city_fraction)))
    held=sorted(np.random.default_rng(a.seed).permutation(cities)[:n].tolist())
    unseen=frame.city.astype(str).isin(held).to_numpy()
    taskcol=frame.indicator_column.astype(str)
    train_ids=np.flatnonzero(taskcol.isin(TRAIN_TASKS).to_numpy() & ~unseen)
    scui=np.flatnonzero(taskcol.eq(HELDOUT_TASK).to_numpy() & ~unseen)
    ucui=np.flatnonzero(taskcol.eq(HELDOUT_TASK).to_numpy() & unseen)
    if min(map(len,(train_ids,scui,ucui)))<1:
        raise ValueError("Missing required source train / Education DEV rows")
    if a.train_limit and a.train_limit<len(train_ids):
        selected=stratified_sample(
            frame.iloc[train_ids].copy().reset_index(),a.train_limit,a.seed)
        train_ids=selected["index"].to_numpy(dtype=np.int64)
    train_frame=frame.iloc[train_ids]
    if set(train_frame.indicator_column)!=set(TRAIN_TASKS):
        raise ValueError("Expected exactly two source indicators")
    stats,unseen_stats=calibrated_moments(
        train_frame,embeddings,HELDOUT_TASK,a.scale_temperature)
    vocab=(*TRAIN_TASKS,HELDOUT_TASK)
    tokens={task:build_tokens(task,embeddings,cards) for task in vocab}
    device=torch.device("cuda")
    source_sem=torch.stack([embeddings[s].float() for s in TRAIN_TASKS]).to(device)
    tsem=torch.stack([embeddings[names[i]].float() for i in train_ids]).to(device)
    edges=torch.stack([tokens[names[i]][0] for i in train_ids]).to(device)
    flags=torch.stack([tokens[names[i]][1] for i in train_ids]).to(device)
    ids=torch.as_tensor(train_ids,dtype=torch.long)
    mean=torch.tensor([stats[names[i]][0] for i in train_ids],dtype=torch.float32)
    std=torch.tensor([stats[names[i]][1] for i in train_ids],dtype=torch.float32)
    targets=((y[ids]-mean)/std).to(device)
    visual=x[ids].to(device)
    context=ctx[ids].to(device)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    model=ConceptLoRAComposer(
        x.shape[1],source_sem.shape[1],hidden=a.hidden,rank=a.rank).to(device)
    # Frozen Qwen relations teach concept-rank selection, not source weights.
    # All concept targets are computed from the two supervised source tasks,
    # and no Education image or label enters gradient updates.
    target_sem=embeddings[HELDOUT_TASK].float().to(device)
    target_rel,target_flag=tokens[HELDOUT_TASK]
    target_rel,target_flag=target_rel.to(device),target_flag.to(device)
    qwen_concepts,clip_concepts=get_concept_anchors(
        a.text_cache_dir,a.text_model,a.remoteclip_model,
        x.shape[1],source_sem.shape[1])
    teacher=rank_targets(
        target_rel,qwen_concepts.to(device),
        temperature=a.teacher_temperature,rank=a.rank).to(device)
    # Only supervised source imagery supplies alignment moments.
    clip_target=visual_concept_targets(
        visual.cpu(),visual.cpu(),clip_concepts).to(device)
    print("[k7-m5] Qwen concept teacher (Population / Food; "
          "residential, settlement, roads, commercial):",flush=True)
    for source,prob in zip(TRAIN_TASKS,teacher.detach().cpu()):
        by_concept=prob.reshape(4,-1).sum(-1).tolist()
        print(f"[k7-m5] {source}: "+
              " ".join(f"{v:.3f}" for v in by_concept),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    rng=torch.Generator(device="cpu").manual_seed(a.seed)
    print(f"[k7-m5] GPU={torch.cuda.get_device_name(0)} "
          f"train={len(train_ids)} SCUI={len(scui)} UCUI={len(ucui)} "
          f"seed=42 trained_models=1",flush=True)
    for epoch in tqdm(range(a.epochs),desc="M5 one concept-KD fit",
                      dynamic_ncols=True):
        model.train()
        perm=torch.randperm(len(train_ids),generator=rng).to(device)
        total=0.
        steps=0
        for start in range(0,len(train_ids),a.batch_size):
            j=perm[start:start+a.batch_size]
            active=torch.ones((len(j),2),dtype=torch.bool,device=device)
            hide=(torch.rand(len(j),generator=rng)<a.self_dropout).to(device)
            active &= ~(flags[j].bool() & hide[:,None])
            opt.zero_grad(set_to_none=True)
            pred,_,_,activation=model(visual[j],context[j],tsem[j],source_sem,edges[j],flags[j],active,return_concepts=True)
            loss=F.mse_loss(pred,targets[j])
            # Same optimizer step, not a second model: direct self supervision
            # makes the two LoRA parameter blocks source-specialized.
            if a.self_loss:
                self_mask=flags[j].bool()
                self_pred=model(visual[j],context[j],tsem[j],source_sem,
                                edges[j],flags[j],self_mask)
                loss=loss+a.self_loss*F.mse_loss(self_pred,targets[j])
            # Target text trains rank gates, visual CLIP pseudo-scores ground ranks.
            _,_,student_gates=model(
                visual[j],context[j],target_sem.expand(len(j),-1),
                source_sem,target_rel.expand(len(j),-1,-1),
                target_flag.expand(len(j),-1),return_aux=True)
            kd=rank_distillation(student_gates,teacher,rank=a.rank)
            align=concept_supervision(activation,clip_target[j],rank=a.rank)
            loss=loss+a.kd_weight*kd+a.concept_weight*align
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
            opt.step()
            total+=float(loss.detach().cpu())
            steps+=1
        if epoch==0 or epoch+1==a.epochs or (epoch+1)%5==0:
            print(f"[k7-m5] epoch={epoch+1}/{a.epochs} total_loss={total/max(steps,1):.3f}",
                  flush=True)
    model.eval()
    correct,flag=tokens[HELDOUT_TASK]
    generic=generic_embeddings(a.text_cache_dir,a.text_model,len(target_sem))
    generic_tensor=torch.stack([generic[s] for s in TRAIN_TASKS])
    edge_choices=relation_conditions(correct,generic_tensor)
    results=[]
    for part,indices in (("SCUI",scui),("UCUI",ucui)):
        print(f"[k7-m5] {part} inference n={len(indices)}",flush=True)
        original=None
        actual=y[indices].numpy()
        for condition,relation in edge_choices.items():
            prediction,metrics=evaluate(
                model,x,ctx,indices,target_sem,source_sem,relation.to(device),
                flag.to(device),unseen_stats[0],unseen_stats[1],actual,
                a.batch_size,device)
            if condition=="correct":
                original=prediction
            metrics["mean_abs_prediction_change"]=float(
                np.mean(np.abs(prediction-original)))
            results.append({"partition":part,"condition":condition,
                            "n":len(indices),**metrics})
    a.output.mkdir(parents=True,exist_ok=True)
    temp=a.output/"metrics.tmp"
    with temp.open("w",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    temp.replace(a.output/"metrics.csv")
    torch.save({"model":model.cpu().state_dict(),
                "source_tasks":TRAIN_TASKS,"heldout_task":HELDOUT_TASK,
                "heldout_cities":held,"source_stats":stats,
                "unseen_stats":unseen_stats},a.output/"model.pt")
    (a.output/"config.json").write_text(json.dumps({
        **{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        "source_tasks":TRAIN_TASKS,"heldout_task":HELDOUT_TASK,
        "heldout_cities":held,"n_train":len(train_ids),
        "target_policy":"Education text used for rank KD; source-only RemoteCLIP concept grounding; no Education labels or images in optimizer",
        "teacher_concept_rank_probs":[[float(v) for v in row] for row in teacher.detach().cpu()],
        "concept_names":["residential","settlement","roads","commercial"]
    },indent=2))
    print(f"{'Split':5} {'Text':9} {'MSE':>7} {'Pearson':>8} {'Ratio':>7} "
          f"{'PopWt':>7} {'FoodWt':>7}",flush=True)
    for r in results:
        print(f"{r['partition']:5} {r['condition']:9} "
              f"{r['mse_log1p']:7.3f} {r['pearson']:8.3f} {r['ratio']:7.3f} "
              f"{r['population_attention']:7.3f} "
              f"{r['food_attention']:7.3f}",flush=True)
    print(f"[k7-m5] COMPLETE {a.output} (one concept-KD model; three inference texts)",flush=True)


if __name__=="__main__":
    main()
