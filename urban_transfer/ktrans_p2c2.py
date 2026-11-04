"""P2-C2: provenance and parameter-identity audit of five INNER K5-A folds.

Read only last.pt metadata/model tensors and the nested source-only CSV.
Never evaluate an outer target or train a new expert. Same named expert
across different folds is not assumed to share frozen parameters.
"""
from itertools import combinations
from pathlib import Path
import numpy as np

def validated_fold(ckpt,pseudo,sources,outer_target,nested_csv):
    """Check metadata against source-only LOTO protocol (no image features)."""
    meta=ckpt['meta']
    model=ckpt['model']
    expected=set(sources)-{pseudo}
    if meta.get('heldout_task')!=pseudo or set(meta.get('source_tasks',()))!=expected:
        raise ValueError('Inner pseudo task/expert membership violated')
    if outer_target==pseudo or outer_target in meta['source_tasks']:
        raise ValueError('Outer target entered inner folds')
    if not meta.get('normalized_experts') or not meta.get('expert_bank'):
        raise ValueError('Not normalized source expert bank')
    if Path(meta['train_csv']).resolve()!=Path(nested_csv).resolve():
        raise ValueError('Checkpoint trained on wrong nested source-only CSV')
    if not isinstance(model,dict):raise ValueError('Missing model state dictionary')
    if 'epoch' not in ckpt or int(ckpt['epoch'])<1:raise ValueError('Checkpoint epoch absent')
    for name in ('source_stats','seed','heldout_cities'):
        if name not in meta:raise ValueError('Missing source-only checkpoint provenance '+name)
    if set(meta['source_stats'])!=expected:
        raise ValueError('Checkpoint source moments mismatch')
    for idx,expert in enumerate(sorted(expected)):
        prefix=f'experts.{idx}.'
        if not any(key.startswith(prefix) for key in model):
            raise ValueError('Missing expert parameter tensors for '+expert)
    return {'pseudo_task':pseudo,
            'checkpoint_epoch':int(ckpt['epoch']),
            'checkpoint_selection':'last.pt',
            'source_tasks':tuple(sorted(expected)),
            'source_expert_count':len(expected),
            'meta_seed':meta['seed'],
            'n_heldout_inner_cities':len(meta['heldout_cities']),
            'inner_heldout_cities':tuple(meta['heldout_cities']),
            'nested_train_csv':str(Path(meta['train_csv']).resolve())}

def tensors_for_expert(ckpt,expert):
    sources=list(ckpt['meta']['source_tasks'])
    idx=sources.index(expert)
    prefix=f'experts.{idx}.'
    return {k[len(prefix):]:v.detach().cpu().numpy().astype(np.float64).reshape(-1)
            for k,v in ckpt['model'].items() if k.startswith(prefix)}

def compare_vectors(a,b):
    if set(a)!=set(b) or not a:raise ValueError('Different expert parameter architectures')
    for key in a:
        if a[key].shape!=b[key].shape or not np.isfinite(a[key]).all() or not np.isfinite(b[key]).all():
            raise ValueError('Non-comparable expert parameters')
    u=np.concatenate([a[k] for k in sorted(a)])
    v=np.concatenate([b[k] for k in sorted(b)])
    delta=float(np.linalg.norm(u-v))
    norm_a=float(np.linalg.norm(u));norm_b=float(np.linalg.norm(v))
    return {'n_parameters':int(u.size),
            'relative_l2_parameter_shift':float(delta/max((norm_a+norm_b)/2,1e-12)),
            'cosine_parameter_similarity':float(u@v/(norm_a*norm_b)) if norm_a*norm_b>1e-12 else float('nan'),
            'identical_parameters':bool(np.array_equal(u,v))}

def shared_tensors(ckpt):
    return {k[len('shared.'):]:v.detach().cpu().numpy().astype(np.float64).reshape(-1)
            for k,v in ckpt['model'].items() if k.startswith('shared.')}

def fold_parameter_audit(checkpoints,sources,outer_target,nested_csv):
    names=tuple(sorted(sources))
    if set(checkpoints)!=set(names):raise ValueError('Missing pseudo task checkpoints')
    metadata={}
    for pseudo in names:
        metadata[pseudo]=validated_fold(checkpoints[pseudo],pseudo,names,outer_target,nested_csv)
    rows=[]
    shared=[]
    for first,second in combinations(names,2):
        left=checkpoints[first];right=checkpoints[second]
        common=sorted((set(names)-{first}) & (set(names)-{second}))
        shared.append({'pseudo_a':first,'pseudo_b':second,
                       **compare_vectors(shared_tensors(left),shared_tensors(right))})
        for expert in common:
            rows.append({'pseudo_a':first,'pseudo_b':second,
                         'named_expert':expert,
                         **compare_vectors(tensors_for_expert(left,expert),
                                           tensors_for_expert(right,expert))})
    return list(metadata.values()),rows,shared

def training_data_audit(folds,nested_csv,sources,outer_target):
    """Reconstruct each fold's supervised source rows from CSV and heldout cities.

    This verifies source task/city memberships, but does not prove
    zero-shot independence from prior checkpoint selection.
    """
    import pandas as pd
    df=pd.read_csv(nested_csv)
    required={'city','indicator_column','image_name'}
    if not required.issubset(df.columns):raise ValueError('Nested CSV lacks identity columns')
    if outer_target in set(df.indicator_column) or set(df.indicator_column)!=set(sources):
        raise ValueError('Nested source CSV contains outer task or missing source tasks')
    identities=list(zip(df['city'].astype(str),df['indicator_column'].astype(str),
                        df['image_name'].astype(str)))
    if len(identities)!=len(set(identities)):
        raise ValueError('Ambiguous nested source row identity')
    per_fold={};report=[]
    for fold in folds:
        pseudo=fold['pseudo_task'];unseen=set(map(str,fold['inner_heldout_cities']))
        if not unseen.issubset(set(df['city'].astype(str))):raise ValueError('Unknown inner heldout city')
        ids={identities[i] for i,row in enumerate(df.itertuples())
             if str(row.indicator_column)!=pseudo and str(row.city) not in unseen}
        train_tasks={x[1] for x in ids}
        if train_tasks!=set(sources)-{pseudo}:raise ValueError('Unexpected supervised training sources')
        per_fold[pseudo]=ids
        for expert in sorted(train_tasks):
            subset={x for x in ids if x[1]==expert}
            report.append({'pseudo_task':pseudo,'named_expert':expert,
                'n_supervised_source_rows':len(subset),
                'n_training_cities':len({x[0] for x in subset})})
    overlap=[]
    for a,b in combinations(sorted(per_fold),2):
        A=per_fold[a];B=per_fold[b]
        common=len(A&B);union=len(A|B)
        overlap.append({'pseudo_a':a,'pseudo_b':b,
            'n_training_rows_a':len(A),'n_training_rows_b':len(B),
            'n_shared_training_rows':common,
            'training_row_jaccard':float(common/union) if union else float('nan')})
    return report,overlap
