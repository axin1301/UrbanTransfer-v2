"""Transfer-oriented auxiliary objectives with explicit group isolation.

Source-only training labels. Pearson is computed within each task; pairwise
order constraints use ONLY pairs sharing both task and city. The losses
operate on standardized labels/predictions and cannot correct task-scale bias.
"""
import torch
import torch.nn.functional as F

def grouped_correlation_loss(pred,target,tasks,min_size=8,epsilon=1e-6):
    """Mean (1-Pearson) across sufficiently populated source-task groups."""
    losses=[]
    for task in sorted(set(tasks)):
        indices=[i for i,t in enumerate(tasks) if t==task]
        if len(indices)<min_size:continue
        p=pred[indices];y=target[indices]
        pc=p-p.mean();yc=y-y.mean()
        # Ignore groups with constant labels to avoid fitting noise.
        denom=torch.linalg.vector_norm(pc)*torch.linalg.vector_norm(yc)
        if float(torch.linalg.vector_norm(yc).detach())<epsilon:continue
        corr=(pc*yc).sum()/(denom+epsilon)
        losses.append(1-corr)
    return torch.stack(losses).mean() if losses else pred.sum()*0.

def city_pairwise_rank_loss(pred,target,tasks,cities,margin=.2,max_pairs=256):
    """Rank only within the same task AND city, with meaningful label gaps.

    Deterministic adjacent pairs after sorting by target label; no random
    pairing, no cross-city/task ranking, and bounded pairs per group.
    """
    if len(tasks)!=len(cities) or len(tasks)!=len(pred):
        raise ValueError('Task, city, prediction lengths must match')
    groups={}
    for i,key in enumerate(zip(tasks,cities)):
        groups.setdefault(key,[]).append(i)
    terms=[]
    for key in sorted(groups):
        ids=groups[key]
        if len(ids)<2:continue
        order=sorted(ids,key=lambda i:float(target[i].detach()))
        # Compare low/high quantile samples, not almost-identical neighbors.
        half=len(order)//2
        if half==0:continue
        low=order[:half];high=order[-half:]
        pairs=list(zip(low,high))[:max_pairs]
        if not pairs:continue
        a=torch.tensor([i for i,j in pairs],device=pred.device)
        b=torch.tensor([j for i,j in pairs],device=pred.device)
        diff=target[b]-target[a]
        valid=diff>margin
        if not bool(valid.any()):continue
        # Normalize logit by fixed target-difference scale. Softplus
        # encourages ordered margins, not just positive sign.
        gap=(pred[b]-pred[a])[valid]
        terms.append(F.softplus(1.-gap).mean())
    return torch.stack(terms).mean() if terms else pred.sum()*0.

def transfer_loss(spatial,label_std,tasks,cities,mode='mse',
                  corr_weight=.2,rank_weight=.1):
    if mode not in ('mse','corr','rank','combined'):
        raise ValueError('Unknown loss mode')
    mse=(spatial-label_std).square().mean()
    corr=(grouped_correlation_loss(spatial,label_std,tasks)
          if mode in ('corr','combined') else mse.detach()*0.)
    rank=(city_pairwise_rank_loss(spatial,label_std,tasks,cities)
          if mode in ('rank','combined') else mse.detach()*0.)
    total=mse
    if mode in ('corr','combined'):total=total+corr_weight*corr
    if mode in ('rank','combined'):total=total+rank_weight*rank
    return total,{'mse':mse.detach(),'corr':corr.detach(),'rank':rank.detach()}
