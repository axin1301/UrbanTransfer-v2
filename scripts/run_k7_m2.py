"""K7 M2: ONE training run, two labelled sources, one unseen Education validation.

Directed source-specific hyper-adapters + image-conditioned token fusion.
The target is never used for training, checkpoint selection or scale fitting.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from scripts.run_k7_m1 import get_cards
from scripts.run_z1_pilot import calibrated_moments, scores
from scripts.run_z1_relation import diagnostic
from urban_transfer.data_m1 import stratified_sample
from urban_transfer.k7_directed_fusion import DirectedEdgeFusionRegressor
from urban_transfer.k7_feature_cache import read_tensor_cache
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS, HELDOUT_TASK
from urban_transfer.ktrans_text import qwen_task_vectors


def build_tokens(target, embeddings, relations):
    """Keep BOTH source edges, with fixed source order and explicit self flag.

    Source->self is a neutral SELF token, not a fabricated LLM relation.
    The other source->training target uses its directed mechanism card.
    At unseen Education both edges use the genuine directed cards.
    """
    tokens, flags = [], []
    for source in TRAIN_TASKS:
        if source == target:
            tokens.append(torch.zeros_like(embeddings[source]).float())
            flags.append(1.)
        else:
            tokens.append(relations[source,target].float())
            flags.append(0.)
    return torch.stack(tokens), torch.tensor(flags,dtype=torch.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--feature-dir", type=Path, default=Path("runs/m3_features"))
    p.add_argument("--tensor-cache", type=Path,
                   default=Path("runs/k7/e1_e2_v1/_source_tensors.pt"))
    p.add_argument("--remoteclip-model", default="ViT-B-32")
    p.add_argument("--text-cache-dir", type=Path, default=Path("runs/ktrans_text"))
    p.add_argument("--text-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--output", type=Path,
                   default=Path("runs/k7/m2_directed_fusion_seed42"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-limit", type=int, default=20000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--city-fraction", type=float, default=0.25)
    p.add_argument("--scale-temperature", type=float, default=0.2)
    p.add_argument("--self-dropout", type=float, default=0.5)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available; run inside an MI250/ROCm allocation")
    if a.seed != 42:
        raise ValueError("M2 intentionally supports only the fixed seed=42")
    if not (0 <= a.self_dropout < 1) or not 0<a.city_fraction<0.5:
        p.error("Invalid dropout / city fraction")
    if a.epochs < 1 or a.batch_size < 1:
        p.error("Invalid epochs or batch size")
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(f"Refusing overwrite of completed M2 run: {a.output}")
    frame, x, ctx, y, names = read_tensor_cache(
        a.tensor_cache,a.source_csv,a.image_root,a.feature_dir,a.remoteclip_model)
    embeddings = qwen_task_vectors(
        root=a.text_cache_dir,model_id=a.text_model,profile="short",only_cached=True)
    cards = get_cards(a.text_cache_dir,a.text_model)
    if set((*TRAIN_TASKS,HELDOUT_TASK)) - set(names):
        raise ValueError("Missing fixed M2 source/target indicators")
    cities = sorted(frame.city.astype(str).unique())
    if len(cities) < 3:
        raise ValueError("Insufficient source cities")
    held_count = max(1,min(len(cities)-2,round(len(cities)*a.city_fraction)))
    held = sorted(np.random.default_rng(a.seed).permutation(cities)[:held_count].tolist())
    unseen = frame.city.astype(str).isin(held).to_numpy()
    tasks = frame.indicator_column.astype(str)
    source_mask = tasks.isin(TRAIN_TASKS).to_numpy()
    val_mask = tasks.eq(HELDOUT_TASK).to_numpy()
    train_ids = np.flatnonzero(source_mask & ~unseen)
    scui = np.flatnonzero(val_mask & ~unseen)
    ucui = np.flatnonzero(val_mask & unseen)
    if min(map(len,(train_ids,scui,ucui))) < 1:
        raise ValueError("Empty source train / unseen target partition")
    if a.train_limit and a.train_limit < len(train_ids):
        chosen = stratified_sample(
            frame.iloc[train_ids].copy().reset_index(),a.train_limit,a.seed)
        train_ids = chosen["index"].to_numpy(dtype=np.int64)
    train_frame = frame.iloc[train_ids]
    if set(train_frame.indicator_column) != set(TRAIN_TASKS):
        raise ValueError("Subsampling removed a source")
    stats,unseen_stats = calibrated_moments(
        train_frame,embeddings,HELDOUT_TASK,a.scale_temperature)
    vocab = (*TRAIN_TASKS,HELDOUT_TASK)
    token_lookup = {name:build_tokens(name,embeddings,cards) for name in vocab}
    device = torch.device("cuda")
    source_sem = torch.stack([embeddings[s].float() for s in TRAIN_TASKS]).to(device)
    tsem = torch.stack([embeddings[names[i]].float() for i in train_ids]).to(device)
    tokens = torch.stack([token_lookup[names[i]][0] for i in train_ids]).to(device)
    flags = torch.stack([token_lookup[names[i]][1] for i in train_ids]).to(device)
    ids = torch.tensor(train_ids,dtype=torch.long)
    mean = torch.tensor([stats[names[i]][0] for i in train_ids],dtype=torch.float32)
    std = torch.tensor([stats[names[i]][1] for i in train_ids],dtype=torch.float32)
    norm_y = ((y[ids]-mean)/std).to(device)
    visual,context = x[ids].to(device),ctx[ids].to(device)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    model = DirectedEdgeFusionRegressor(
        x.shape[1],source_sem.shape[1],hidden=a.hidden,rank=a.rank).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    rng = torch.Generator(device="cpu").manual_seed(a.seed)
    print(f"[k7-m2] GPU={torch.cuda.get_device_name(0)} "
          f"train={len(train_ids)} SCUI={len(scui)} UCUI={len(ucui)} "
          f"held_cities={len(held)} seed={a.seed}",flush=True)
    # The two tokens are present throughout training. On half the rows,
    # mask the target's SELF expert so gradients train the cross-source path.
    progress = tqdm(range(a.epochs),desc="M2 single directed-fusion model",
                    dynamic_ncols=True)
    for epoch in progress:
        model.train()
        perm = torch.randperm(len(train_ids),generator=rng).to(device)
        total = 0.
        steps = 0
        for start in range(0,len(train_ids),a.batch_size):
            j = perm[start:start+a.batch_size]
            active = torch.ones((len(j),2),dtype=torch.bool,device=device)
            hide_self = (torch.rand(len(j),device="cpu",generator=rng)
                         < a.self_dropout).to(device)
            active &= ~(flags[j].bool() & hide_self.unsqueeze(-1))
            opt.zero_grad(set_to_none=True)
            z = model(visual[j],context[j],tsem[j],source_sem,
                      tokens[j],flags[j],active)
            loss = F.mse_loss(z,norm_y[j])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step()
            total += float(loss.detach().cpu())
            steps += 1
        progress.set_postfix(loss=f"{total/max(1,steps):.3f}")
    model.eval()
    target_s = embeddings[HELDOUT_TASK].float().to(device)
    target_rel,target_flags = token_lookup[HELDOUT_TASK]
    target_rel = target_rel.to(device)
    target_flags = target_flags.to(device)
    output = []
    # Access education labels only AFTER training and prediction.
    for part,indices in (("SCUI",scui),("UCUI",ucui)):
        preds,weights = [],[]
        with torch.inference_mode():
            for start in tqdm(range(0,len(indices),a.batch_size),
                              desc=f"M2 {part} inference",leave=False,
                              dynamic_ncols=True):
                j = torch.tensor(indices[start:start+a.batch_size],dtype=torch.long)
                z,w = model(x[j].to(device),ctx[j].to(device),
                            target_s.expand(len(j),-1),source_sem,
                            target_rel.expand(len(j),-1,-1),
                            target_flags.expand(len(j),-1),
                            return_attention=True)
                preds.append(z.cpu())
                weights.append(w.cpu())
        pred = unseen_stats[0]+unseen_stats[1]*torch.cat(preds).numpy()
        actual = y[indices].numpy()
        summary = scores(pred,actual)
        baseline = float(np.mean((unseen_stats[0]-actual)**2))
        attn = torch.cat(weights).mean(0).numpy()
        row = {"partition":part, **summary,
               "source_mean_mse":baseline,
               "relative_mse_to_source_mean":summary["mse_log1p"]/max(baseline,1e-12),
               "population_attention":float(attn[0]),
               "food_attention":float(attn[1]),
               **diagnostic(pred,actual)}
        output.append(row)
        print(f"[k7-m2] {part} MSE={summary['mse_log1p']:.3f} "
              f"Pearson={summary['pearson']:.3f} "
              f"Ratio={row['relative_mse_to_source_mean']:.3f} "
              f"PopAttn={attn[0]:.3f} FoodAttn={attn[1]:.3f}",flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/"metrics.csv").open("w",newline="") as fh:
        writer = csv.DictWriter(fh,fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    torch.save({"model":model.cpu().state_dict(),"source_tasks":TRAIN_TASKS,
                "heldout_task":HELDOUT_TASK,"heldout_cities":held,
                "source_stats":stats,"unseen_stats":unseen_stats},
               a.output/"model.pt")
    (a.output/"config.json").write_text(json.dumps({
        **{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        "source_tasks":TRAIN_TASKS,"heldout_task":HELDOUT_TASK,
        "heldout_cities":held,"n_train":len(train_ids),
        "note":"M2 fixed source-only DEV; no true target test files."
    },indent=2))
    print(f"[k7-m2] COMPLETE {a.output}",flush=True)


if __name__ == "__main__":
    main()
