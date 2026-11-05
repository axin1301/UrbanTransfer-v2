"""K7 E3: cross-fitted source transfer rewards for directed relation routing.

Inner-city labels of ONLY the five outer-source indicators supervise utility.
Use the existing K7-E1 model as an adapter bank; no target pseudo-task
labels or outer unseen-city labels ever enter fitting.
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from urban_transfer.k7_hyper import K7HyperRegressor


def pair_features(source, target, semantics, relations, sources, shuffle=False):
    choices = tuple(sorted(sources))
    if source == target or source not in choices:
        raise ValueError("Invalid directed source->target pair")
    relation_source = choices[(choices.index(source)+1) % len(choices)] if shuffle else source
    a = F.normalize(semantics[target].float(), dim=0)
    b = F.normalize(semantics[source].float(), dim=0)
    r = F.normalize(relations[target, relation_source].float(), dim=0)
    return np.array([1., float(a@b), float(a@r), float(b@r),
                     float((a*b)@r), float(torch.linalg.vector_norm(a-b))],
                    dtype=np.float64)


def fit_ridge(examples, rewards, alpha=1.0):
    x = np.asarray(examples, dtype=np.float64)
    y = np.asarray(rewards, dtype=np.float64)
    if x.ndim != 2 or len(x) < x.shape[1] or len(y) != len(x):
        raise ValueError("Insufficient utility training pairs")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite utility features/rewards")
    penalty = alpha * np.eye(x.shape[1])
    penalty[0,0] = 0.
    return np.linalg.solve(x.T@x+penalty, x.T@y)


def inner_city_split(frame, outer_train, seed, fraction=0.2):
    outer = frame.iloc[outer_train]
    cities = sorted(outer.city.astype(str).unique())
    if len(cities) < 3:
        raise ValueError("Need 3+ source train cities for reward split")
    count = max(1, min(len(cities)-2, round(len(cities)*fraction)))
    chosen = set(np.random.default_rng(seed+2027).permutation(cities)[:count])
    mask = outer.city.astype(str).isin(chosen).to_numpy()
    train, val = outer_train[~mask], outer_train[mask]
    if set(frame.iloc[train].indicator_column) != set(frame.iloc[outer_train].indicator_column):
        raise ValueError("Inner-city split dropped a source indicator")
    return train, val


def predict_adapter(model, x, ctx, ids, semantic, batch_size, device):
    results = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            j = torch.as_tensor(ids[start:start+batch_size], dtype=torch.long, device=device)
            text = semantic.to(device).expand(len(j), -1)
            results.append(model(x[j], ctx[j], text).cpu())
    return torch.cat(results).numpy()


def fit_inner(a, frame, x, ctx, y, names, train_ids, semantics, device):
    """Learn an E1 model on inner TRAIN cities, never on reward city labels."""
    tasks = sorted(set(names[i] for i in train_ids))
    moments = {}
    for t in tasks:
        vals = np.asarray([float(y[i]) for i in train_ids if names[i] == t])
        moments[t] = (float(vals.mean()), max(float(vals.std()), 1e-3))
    ids = torch.as_tensor(train_ids, dtype=torch.long, device=device)
    text = torch.stack([semantics[names[i]] for i in train_ids]).to(device)
    target = torch.tensor([(float(y[i])-moments[names[i]][0])/moments[names[i]][1]
                           for i in train_ids], dtype=torch.float32, device=device)
    torch.manual_seed(a.seed)
    model = K7HyperRegressor(x.shape[1], text.shape[1], a.hidden, a.rank, False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    epochs = tqdm(range(a.epochs), desc="K7 inner utility fit", dynamic_ncols=True, leave=False)
    for _ in epochs:
        model.train()
        permutation = torch.randperm(len(ids), generator=gen).to(device)
        total = 0.
        steps = 0
        for start in range(0, len(ids), a.batch_size):
            j = permutation[start:start+a.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x[ids[j]], ctx[ids[j]], text[j]), target[j])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach().cpu())
            steps += 1
        epochs.set_postfix(loss=f"{total/max(1,steps):.3f}")
    return model.eval(), moments


def measured_utilities(a, model, frame, x, ctx, y, names, val_ids, moments, semantics, device):
    """Target-own-adapter MSE minus source-adapter MSE, on inner cities only."""
    utilities = {}
    for target in sorted(moments):
        ids = np.asarray([i for i in val_ids if names[i] == target], dtype=int)
        if len(ids) < 2:
            raise ValueError("No inner reward validation for task " + target)
        if len(ids) > a.reward_max_rows:
            ids = np.random.default_rng(a.seed+len(target)).choice(
                ids, size=a.reward_max_rows, replace=False)
        observed = (y[ids].numpy()-moments[target][0])/moments[target][1]
        direct = predict_adapter(model, x, ctx, ids, semantics[target], a.batch_size, device)
        direct_mse = float(np.mean((direct-observed)**2))
        for source in sorted(moments):
            if target == source:
                continue
            source_pred = predict_adapter(model, x, ctx, ids, semantics[source],
                                          a.batch_size, device)
            utilities[target, source] = direct_mse-float(np.mean((source_pred-observed)**2))
    return utilities


def predicted_weights(target, source_names, utilities, semantics, relations, shuffle=False,
                      alpha=1.0):
    """Fit rewards on source->source pairs; infer held-target source weights."""
    x = []
    y = []
    for (seen_target, source), reward in sorted(utilities.items()):
        candidates = [s for s in source_names if s != seen_target]
        x.append(pair_features(source, seen_target, semantics, relations, candidates, shuffle))
        y.append(reward)
    coefficients = fit_ridge(x, y, alpha)
    feats = np.stack([pair_features(s, target, semantics, relations,
                                   source_names, shuffle) for s in source_names])
    predicted = feats @ coefficients
    # Zero-utility direct target adapter is an explicit no-transfer option.
    logits = np.r_[0., predicted] / 0.25
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return weights, predicted
