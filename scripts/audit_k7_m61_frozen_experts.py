"""Read-only M6.1 expert and same-image indicator interventions.

Never fits models, updates checkpoints, invokes Qwen-VL or accesses final
test files. Education pseudo-dev labels are used for REPORTING only.
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

from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.k7_mllm_hidden import HiddenStore,TASKS
from urban_transfer.k7_m61_router import StructuredMLLMRouter
from scripts.run_k7_m61_student import load_precomputed
from scripts.run_z1_pilot import scores


@torch.inference_mode()
def predict_interventions(model,visual,context,tokens):
    """Force mixing weights BEFORE the nonlinear shared regression readout.

    Reproduces the original router output exactly; avoids the incorrect
    approximation of mixing post-readout expert predictions.
    """
    knowledge=model.read_hidden(tokens)
    h=model.visual(torch.cat([visual.float(),context.float()],dim=-1))
    expert=torch.stack([module(h) for module in model.experts],dim=1)
    expert=expert*model.gate(knowledge)[:,None,:]
    learned=F.softmax(model.router(knowledge),dim=-1)
    forced={
        "learned":learned,
        "population_only":torch.tensor([1.,0.],device=learned.device).expand_as(learned),
        "food_only":torch.tensor([0.,1.],device=learned.device).expand_as(learned),
        "equal_mix":torch.full_like(learned,.5),
    }
    predictions={}
    for name,weights in forced.items():
        features=(expert*weights[:,:,None]).sum(dim=1)
        predictions[name]=model.out(features).squeeze(-1)
    return predictions,learned,knowledge


def aggregate(values):
    arr=np.asarray(values,dtype=np.float64)
    if arr.size==0 or not np.isfinite(arr).all():
        raise ValueError("Empty/nonfinite diagnostic array")
    return {"mean":float(arr.mean()),"std":float(arr.std()),
            "min":float(arr.min()),"max":float(arr.max())}


def validate_partition(frame,manifest):
    train=np.asarray(manifest["train"],dtype=np.int64)
    scui=np.asarray(manifest["SCUI"],dtype=np.int64)
    ucui=np.asarray(manifest["UCUI"],dtype=np.int64)
    all_ix=np.concatenate([train,scui,ucui])
    if len(set(map(int,all_ix)))!=len(all_ix):
        raise ValueError("M6.1 audit folds overlap")
    task=frame.indicator_column.astype(str).to_numpy()
    unseen=frame.city.astype(str).isin(manifest["held_cities"]).to_numpy()
    if not (np.isin(task[train],TRAIN_TASKS).all() and
            not unseen[train].any() and
            np.all(task[scui]==HELDOUT_TASK) and not unseen[scui].any() and
            np.all(task[ucui]==HELDOUT_TASK) and unseen[ucui].all()):
        raise ValueError("M6.1 audit violates source-only / city-isolated split")
    return train,scui,ucui


@torch.inference_mode()
def dev_expert_scores(model,features,context,truth,indices,hidden_cache,
                      unseen_mean,unseen_std,batch_size,device):
    all_predictions={key:[] for key in
                     ("learned","population_only","food_only","equal_mix")}
    weights=[]
    for start in range(0,len(indices),batch_size):
        ix=indices[start:start+batch_size]
        hs=torch.stack([hidden_cache[int(i),HELDOUT_TASK] for i in ix]).to(device)
        predictions,route,_=predict_interventions(
            model,features[ix].to(device),context[ix].to(device),hs)
        for name,pred in predictions.items():
            all_predictions[name].append(pred.cpu().numpy())
        weights.append(route[:,0].cpu().numpy())
    route=np.concatenate(weights)
    targets=truth[torch.as_tensor(indices,dtype=torch.long)].numpy()
    report={}
    for name,values in all_predictions.items():
        prediction=unseen_mean+unseen_std*np.concatenate(values)
        report[name]=scores(prediction,targets)
    report["population_router_weight"]=aggregate(route)
    # Include exact baseline for interpretation, without selecting a model.
    report["source_only_constant_baseline_mse"]=float(
        np.mean((unseen_mean-targets)**2))
    return report


@torch.inference_mode()
def same_image_task_test(model,features,context,indices,hidden_cache,
                         batch_size,device):
    weights={task:[] for task in TASKS}
    predictions={task:[] for task in TASKS}
    representations={task:[] for task in TASKS}
    for start in range(0,len(indices),batch_size):
        ix=indices[start:start+batch_size]
        x=features[ix].to(device)
        c=context[ix].to(device)
        for task in TASKS:
            h=torch.stack([hidden_cache[int(i),task] for i in ix]).to(device)
            out,w,k=predict_interventions(model,x,c,h)
            predictions[task].append(out["learned"].cpu().numpy())
            weights[task].append(w[:,0].cpu().numpy())
            representations[task].append(k.cpu())
    w={k:np.concatenate(v) for k,v in weights.items()}
    p={k:np.concatenate(v) for k,v in predictions.items()}
    r={k:F.normalize(torch.cat(v).float(),dim=-1) for k,v in
       representations.items()}
    comparisons={}
    for left,right in ((TRAIN_TASKS[0],HELDOUT_TASK),
                       (TRAIN_TASKS[1],HELDOUT_TASK),
                       (TRAIN_TASKS[0],TRAIN_TASKS[1])):
        key=left+"__vs__"+right
        comparisons[key]={
            "abs_population_router_weight_change":aggregate(np.abs(w[left]-w[right])),
            "abs_standardized_prediction_change":aggregate(np.abs(p[left]-p[right])),
            "knowledge_cosine_similarity":aggregate(
                (r[left]*r[right]).sum(-1).numpy()),
        }
    return {"per_indicator_population_weight":{
                task:aggregate(weights) for task,weights in w.items()},
            "same_image_pairwise":comparisons,
            "caveat":"Same RemoteCLIP image and geo; changing text prompt also changes joint MLLM computation; sensitivity does not prove correct causal knowledge transfer."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,
                        default=Path("runs/k7/m61_structured_seed42"))
    parser.add_argument("--output",type=Path,default=None)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    a=parser.parse_args()
    if a.batch_size<1:parser.error("batch size must be positive")
    manifest=json.loads((a.run_dir/"manifest.json").read_text())
    config=json.loads((a.run_dir/"config.json").read_text())
    if int(config["seed"])!=42 or tuple(config["source_tasks"])!=TRAIN_TASKS or (
        config["heldout_task"]!=HELDOUT_TASK or
        list(config["group_counts"])!=list(manifest["group_counts"])):
        raise ValueError("Incompatible M6.1 configuration/checkpoint")
    output=a.output or a.run_dir/"frozen_expert_indicator_audit.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing audit: "+str(output))
    if a.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm GPU requested but unavailable")

    frame,x,ctx,y,names=read_tensor_cache(
        Path(manifest["tensor_cache"]),Path(manifest["source_csv"]),
        Path(manifest["image_root"]),Path(manifest["feature_dir"]),
        manifest["remoteclip_model"])
    train,scui,ucui=validate_partition(frame,manifest)
    ds=ImageUrbanDataset(manifest["source_csv"],manifest["image_root"],
                         partition="eval")
    if len(ds.df)!=len(frame) or ds.df.indicator_column.astype(str).tolist()!=names:
        raise ValueError("Feature/cache image row alignment mismatch")
    store=HiddenStore(a.run_dir/"hidden.sqlite3")
    source_cache=load_precomputed(frame,train,TASKS,ds,store,manifest)
    eval_ids=np.unique(np.concatenate((scui,ucui)))
    target_cache=load_precomputed(frame,eval_ids,(HELDOUT_TASK,),ds,store,manifest)

    checkpoint=torch.load(a.run_dir/"model.pt",map_location="cpu",weights_only=True)
    mean,std=checkpoint["unseen_stats"]
    if list(checkpoint["heldout_cities"])!=list(manifest["held_cities"]):
        raise ValueError("Checkpoint city split differs from cache manifest")
    first_hidden=source_cache[int(train[0]),HELDOUT_TASK]
    model=StructuredMLLMRouter(
        x.shape[1],first_hidden.shape[1],hidden=int(config["hidden"]),
        heads=int(config["heads"]),group_counts=tuple(manifest["group_counts"]))
    model.load_state_dict(checkpoint["model"],strict=True)
    model=model.to(a.device).eval()
    target_results={}
    for split,ids in (("SCUI",scui),("UCUI",ucui)):
        target_results[split]=dev_expert_scores(
            model,x,ctx,y,ids,target_cache,mean,std,a.batch_size,a.device)
    source_diagnostic=same_image_task_test(
        model,x,ctx,train,source_cache,a.batch_size,a.device)
    report={
        "protocol":"Frozen M6.1 checkpoint, no optimizer, no MLLM inference; source train same-image task comparisons, Education pseudo-dev labels REPORT ONLY; never official tests",
        "train_n":len(train),"scui_n":len(scui),"ucui_n":len(ucui),
        "target_expert_interventions":target_results,
        "source_same_image_indicators":source_diagnostic,
        "warning":"Expert interventions share nonlinearity downstream of the mixture; results are inference-only counterfactuals, not retrained experts.",
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(f"[k7-m61-audit] source={len(train)} SCUI={len(scui)} UCUI={len(ucui)}")
    for split,group in target_results.items():
        print(f"[k7-m61-audit] {split}")
        for mode in ("learned","population_only","food_only","equal_mix"):
            record=group[mode]
            print(f"  {mode:17s} MSE={record['mse_log1p']:.3f} "
                  f"Pearson={record['pearson']:.3f}")
        route=group["population_router_weight"]
        print(f"  PopWt mean={route['mean']:.3f} std={route['std']:.3f}")
    print("[k7-m61-audit] Same-image, changed-indicator prompts (SOURCE TRAIN ONLY):")
    for key,values in source_diagnostic["same_image_pairwise"].items():
        print(f"  {key}: "
              f"abs PopWt delta={values['abs_population_router_weight_change']['mean']:.4f} "
              f"abs standardized pred delta={values['abs_standardized_prediction_change']['mean']:.4f} "
              f"knowledge cosine={values['knowledge_cosine_similarity']['mean']:.4f}")
    print(f"[k7-m61-audit] saved {output}")


if __name__=="__main__":
    main()
