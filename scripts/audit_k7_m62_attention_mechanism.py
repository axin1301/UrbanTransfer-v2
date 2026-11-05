"""Read-only M6.2 attention mechanism + task-query bypass audit.

Never trains, encodes images, modifies a checkpoint/cache, or reads
official held-out test CSVs. Education pseudo-dev labels used for
REPORTING only. Same-image Pop/Food/Education attention comparisons
are computed on source-training tiles only.
"""
import argparse
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.run_k7_m61_student import load_precomputed
from scripts.run_k7_m62_task_query import validate_split
from scripts.run_z1_pilot import scores
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.k7_mllm_hidden import HiddenStore,TASKS
from urban_transfer.k7_m62_task_query import TaskQueriedRouter


MODES=("normal","zero_query_direct","zero_fused_direct","zero_attention_output")


def distributions_stats(x):
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Empty or nonfinite diagnostic values")
    return {"mean":float(x.mean()),"std":float(x.std()),
            "min":float(x.min()),"max":float(x.max())}


def attention_diagnostics(model,tokens):
    """Exact learned-query cross attention with *per-head* attention weights.

    The model's MultiheadAttention module and trainable parameters are
    reused; no copy, alternate projection, or reweighting.
    """
    visual,definition,relation=model.split(tokens.float())
    d=model.definition_tokens(definition).mean(1)
    r=model.relation_tokens(relation).mean(1)
    query=model.task_query(torch.cat([d,r],dim=-1))
    visual=model.image_tokens(visual)
    attended,weights=model.cross_attention(
        query.unsqueeze(1),visual,visual,
        need_weights=True,average_attn_weights=False)
    # [B,heads,one-query,Nvisual], [B,D]
    if (weights.ndim!=4 or weights.shape[2]!=1 or
        weights.shape[3]!=model.group_counts[0]):
        raise ValueError("Unexpected attention weights shape")
    attended=attended.squeeze(1)
    fused=model.fusion(torch.cat(
        [query,attended,query*attended],dim=-1))
    return query,attended,fused,weights.squeeze(2)


def masked_forward(model,visual,context,query,fused,attended,mode):
    """Ablate one direct path, holding other precomputed branches fixed.

    zero_query_direct changes only the explicit direct q -> router/gate
    concatenation; fused(q,attention) remains exactly as originally
    computed. zero_fused_direct removes the other concatenation half.
    zero_attention_output recomputes the fusion from q and zero
    attended; keeps q -> direct router/gate path intact.
    """
    if mode=="normal":
        pass
    elif mode=="zero_query_direct":
        query=torch.zeros_like(query)
    elif mode=="zero_fused_direct":
        fused=torch.zeros_like(fused)
    elif mode=="zero_attention_output":
        # query is untouched, so this isolates attention output through
        # fusion while keeping task-query direct path.
        fused=model.fusion(torch.cat([
            query,torch.zeros_like(attended),
            torch.zeros_like(attended)],dim=-1))
    else:
        raise ValueError("Unknown mechanism intervention")
    cond=torch.cat([query,fused],dim=-1)
    h=model.visual(torch.cat([visual.float(),context.float()],dim=-1))
    expert=torch.stack([part(h) for part in model.experts],dim=1)
    expert=expert*model.gate(cond).unsqueeze(1)
    route=F.softmax(model.router(cond),dim=-1)
    pred=model.out((expert*route.unsqueeze(-1)).sum(dim=1)).squeeze(-1)
    return pred,route


def normalized_entropy(attention):
    """Per sample, mean normalized head entropy (0 concentrated, 1 uniform)."""
    if attention.ndim!=3 or attention.shape[-1]<2:
        raise ValueError("Expected [B,heads,n_visual>=2]")
    prob=attention.float().clamp_min(1e-12)
    prob=prob/prob.sum(-1,keepdim=True)
    return (-(prob*prob.log()).sum(-1)/math.log(attention.shape[-1])).mean(-1)


def js_divergence(p,q):
    """Symmetric Jensen-Shannon divergence (nats), averaged over heads."""
    if p.shape!=q.shape or p.ndim!=3:
        raise ValueError("Expected equally shaped [B,heads,Nvisual]")
    p=p.float().clamp_min(1e-12)
    q=q.float().clamp_min(1e-12)
    p=p/p.sum(-1,keepdim=True)
    q=q/q.sum(-1,keepdim=True)
    m=(p+q)*.5
    return (.5*((p*(p.log()-m.log())).sum(-1)+
                (q*(q.log()-m.log())).sum(-1))).mean(-1)


@torch.inference_mode()
def partition_interventions(model,x,ctx,y,indices,cache,mean,scale,
                            batch_size,device):
    if len(indices)<2 or batch_size<1 or scale<=0:
        raise ValueError("Invalid partition/batch/target scale")
    data={mode:{"pred":[],"pop":[]} for mode in MODES}
    entropy=[]
    for start in range(0,len(indices),batch_size):
        ix=indices[start:start+batch_size]
        tokens=torch.stack([cache[int(i),HELDOUT_TASK] for i in ix]).to(device)
        visual=x[ix].to(device)
        geo=ctx[ix].to(device)
        query,attended,fused,attention=attention_diagnostics(model,tokens)
        entropy.append(normalized_entropy(attention).cpu().numpy())
        for mode in MODES:
            pred,weights=masked_forward(
                model,visual,geo,query,fused,attended,mode)
            data[mode]["pred"].append(pred.cpu().numpy())
            data[mode]["pop"].append(weights[:,0].cpu().numpy())
        # Architectural parity check: make sure the audit's normal
        # path exactly reconstructs original eval-mode predictions.
        baseline,w,task,fused_native=model(
            visual,geo,tokens,return_aux=True)
        ref=data["normal"]["pred"][-1]
        if (not np.allclose(ref,baseline.cpu().numpy(),atol=1e-5,rtol=1e-5)
            or not torch.allclose(w[:,0],
                   torch.as_tensor(data["normal"]["pop"][-1],device=w.device),
                   atol=1e-5,rtol=1e-5)
            or not torch.allclose(task,query,atol=1e-5,rtol=1e-5)
            or not torch.allclose(fused_native,fused,atol=1e-5,rtol=1e-5)):
            raise RuntimeError(
                "Mechanism reconstruction disagrees with frozen model")
    truths=y[torch.as_tensor(indices,dtype=torch.long)].numpy()
    anchor=np.concatenate(data["normal"]["pred"])
    anchor_pop=np.concatenate(data["normal"]["pop"])
    base_mse=float(np.mean((mean-truths)**2))
    result={
        "attention_entropy_normalized":distributions_stats(np.concatenate(entropy)),
        "source_mean_baseline_mse":base_mse,
        "modes":{},
    }
    for mode in MODES:
        prediction=np.concatenate(data[mode]["pred"])
        pop=np.concatenate(data[mode]["pop"])
        metric=scores(mean+scale*prediction,truths)
        result["modes"][mode]={
            **metric,
            "ratio":metric["mse_log1p"]/max(base_mse,1e-12),
            "population_weight":distributions_stats(pop),
            "abs_prediction_change_log1p":distributions_stats(
                scale*np.abs(prediction-anchor)),
            "abs_router_change":distributions_stats(np.abs(pop-anchor_pop)),
        }
    return result


@torch.inference_mode()
def source_attention_pairs(model,indices,cache,batch_size,device):
    """Same source images: compare all task definitions/relations.

    Also compare distinct source images under the SAME indicator.
    Comparison here is per visual token position, not a claim of
    geographically aligned pixel-level attribution.
    """
    attn={t:[] for t in TASKS}
    entropy={t:[] for t in TASKS}
    for start in range(0,len(indices),batch_size):
        ix=indices[start:start+batch_size]
        for task in TASKS:
            h=torch.stack([cache[int(i),task] for i in ix]).to(device)
            _,_,_,w=attention_diagnostics(model,h)
            attn[task].append(w.cpu())
            entropy[task].append(normalized_entropy(w).cpu())
    w={t:torch.cat(chunks,dim=0) for t,chunks in attn.items()}
    report={
        "entropy_by_indicator":{
            t:distributions_stats(torch.cat(entropy[t]).numpy())
            for t in TASKS},
        "same_image_cross_indicator_js":{},
        "different_image_same_indicator_js":{},
    }
    for a,b in ((TRAIN_TASKS[0],HELDOUT_TASK),
                (TRAIN_TASKS[1],HELDOUT_TASK),
                (TRAIN_TASKS[0],TRAIN_TASKS[1])):
        key=a+"__vs__"+b
        report["same_image_cross_indicator_js"][key]=distributions_stats(
            js_divergence(w[a],w[b]).numpy())
    if len(indices)<2:
        raise ValueError("Need at least two source images for image variation")
    for task in TASKS:
        report["different_image_same_indicator_js"][task]=distributions_stats(
            js_divergence(w[task],w[task].roll(1,0)).numpy())
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,
                   default=Path("runs/k7/m62_task_query_seed42"))
    p.add_argument("--cache-dir",type=Path,
                   default=Path("runs/k7/m61_structured_seed42"))
    p.add_argument("--output",type=Path,default=None)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=p.parse_args()
    if args.batch_size<1:p.error("batch-size must be positive")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU unavailable")
    output=args.output or args.run_dir/"frozen_mechanism_audit.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing diagnostic: "+str(output))
    config=json.loads((args.run_dir/"config.json").read_text())
    manifest=json.loads((args.cache_dir/"manifest.json").read_text())
    ckpt=torch.load(args.run_dir/"model.pt",map_location="cpu",weights_only=True)
    if (int(config["seed"])!=42 or
        tuple(config["source_tasks"])!=TRAIN_TASKS or
        config["heldout_task"]!=HELDOUT_TASK or
        Path(config["cache_dir"]).resolve()!=args.cache_dir.resolve() or
        list(ckpt["group_counts"])!=list(manifest["group_counts"]) or
        list(ckpt["heldout_cities"])!=list(manifest["held_cities"])):
        raise ValueError("M6.2 checkpoint/cache split mismatch")
    frame,x,ctx,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    train,scui,ucui=validate_split(frame,manifest)
    ds=ImageUrbanDataset(manifest["source_csv"],manifest["image_root"],
                         partition="eval")
    if len(ds.df)!=len(frame) or ds.df.indicator_column.astype(str).tolist()!=names:
        raise ValueError("Source dataset/visual row order mismatch")
    store=HiddenStore(args.cache_dir/"hidden.sqlite3")
    source_cache=load_precomputed(frame,train,TASKS,ds,store,manifest)
    dev=np.unique(np.concatenate((scui,ucui)))
    dev_cache=load_precomputed(frame,dev,(HELDOUT_TASK,),ds,store,manifest)
    example=source_cache[int(train[0]),HELDOUT_TASK]
    model=TaskQueriedRouter(
        x.shape[-1],example.shape[-1],hidden=int(config["hidden"]),
        heads=int(config["heads"]),group_counts=tuple(manifest["group_counts"]))
    model.load_state_dict(ckpt["model"],strict=True)
    model=model.to(args.device).eval()
    mean,scale=map(float,ckpt["target_stats"])
    dev_report={
        part:partition_interventions(model,x,ctx,y,indices,dev_cache,
                                     mean,scale,args.batch_size,args.device)
        for part,indices in (("SCUI",scui),("UCUI",ucui))}
    source_report=source_attention_pairs(
        model,train,source_cache,args.batch_size,args.device)
    result={
        "protocol":"Frozen M6.2; SCUI/UCUI labels used only for reporting; source train images for same-image cross-indicator JS. No optimizer, no MLLM, no final test.",
        "n_source":len(train),"n_scui":len(scui),"n_ucui":len(ucui),
        "n_visual":manifest["group_counts"][0],
        "modes":{
            "zero_query_direct":"Zeros ONLY explicit task query half of [q,fused]. The fused branch still contains task info.",
            "zero_fused_direct":"Zeros ONLY fused half of [q,fused]. Explicit query remains.",
            "zero_attention_output":"Recomputes fused(q,0,0), retains explicit q and model nonlinearity. Off manifold.",
        },
        "caveat":"MHA weight variations are not causal spatial explanations; Qwen visual tokens are not necessarily geographic pixel patches. Zero ablation shifts representations out of distribution. Source text-conditioned hidden may contain image signal.",
        "dev_mechanism_interventions":dev_report,
        "source_attention_distributions":source_report,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(f"[k7-m62-mechanism] frozen=1 source={len(train)} "
          f"SCUI={len(scui)} UCUI={len(ucui)} "
          f"visual_tokens={manifest['group_counts'][0]}")
    for split,record in dev_report.items():
        print(f"[k7-m62-mechanism] {split} "
              f"normalized_head_entropy={record['attention_entropy_normalized']['mean']:.4f}")
        for mode in MODES:
            item=record["modes"][mode]
            print(f"  {mode:22s} MSE={item['mse_log1p']:.3f} "
                  f"Pearson={item['pearson']:.3f} "
                  f"Ratio={item['ratio']:.3f} "
                  f"|PredDelta|={item['abs_prediction_change_log1p']['mean']:.5f} "
                  f"|PopWtDelta|={item['abs_router_change']['mean']:.5f}")
    print("[k7-m62-mechanism] SOURCE attention JS divergences (nats):")
    for pair,stats in source_report["same_image_cross_indicator_js"].items():
        print(f"  same image {pair}: JS={stats['mean']:.6f}")
    for task,stats in source_report["different_image_same_indicator_js"].items():
        print(f"  different image {task}: JS={stats['mean']:.6f}")
    print(f"[k7-m62-mechanism] saved {output}")


if __name__=="__main__":
    main()
