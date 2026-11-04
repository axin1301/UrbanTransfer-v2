"""Auditable source-only empirical expert competence row assembly.

Inner K5-A pseudo-heldout task checkpoints are evaluated only on
that pseudo-task's inner meta_SCUI / meta_UCUI. Outer target rows,
heldout cities, or K4 outer-best predictions must never be included.
These labels belong to SOURCE pseudo-tasks and are permissible as
conditional-competence supervision, NOT independent final validation.
"""
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from urban_transfer.ktrans_k7d import CUES

def identity(path,task):
    return (str(Path(path).resolve()),str(task))

def manifest_source_index(manifest):
    outer=manifest['heldout_task']
    result={}
    for row in manifest['rows']:
        if row['partition']!='train':continue
        if row['task']==outer:raise ValueError('Outer heldout label in source manifest train')
        key=identity(row['image_path'],row['task'])
        if key in result:raise ValueError('Duplicated image/task source manifest row')
        result[key]=row
    if not result:raise ValueError('Empty outer source manifest rows')
    return result

def validate_inner_fold(pseudo,inner_meta,inner_ds,inner_splits,inner_bank,outer_train_keys,
                        outer_source_tasks,outer_task,nested_csv):
    sources=set(outer_source_tasks)
    if pseudo not in sources or pseudo==outer_task:
        raise ValueError('Invalid pseudo task')
    if inner_meta['heldout_task']!=pseudo or set(inner_bank.source_tasks)!=sources-{pseudo}:
        raise ValueError('Inner experts include pseudo-target or unexpected source')
    if outer_task in set(inner_bank.source_tasks):
        raise ValueError('Outer heldout expert in inner model')
    if Path(inner_meta['train_csv']).resolve()!=Path(nested_csv).resolve():
        raise ValueError('Inner model trained on unexpected source dataset')
    if inner_meta.get('normalized_experts') is not True or inner_meta.get('expert_bank') is not True:
        raise ValueError('Inner fold must use normalized source expert bank')
    known=set();partitions={}
    for split in ('train','meta_SCUI','meta_UCSI','meta_UCUI'):
        indices=inner_splits[split]
        selection=set()
        for ix in indices:
            key=identity(inner_ds.base.paths[int(ix)],inner_ds.df.iloc[int(ix)].indicator_column)
            if key not in outer_train_keys:
                raise ValueError('Inner fold contains a row outside original OUTER TRAIN')
            if key in known:raise ValueError('Duplicate image/task in inner split')
            known.add(key)
            selection.add(key)
            task=key[1]
            if split in ('meta_SCUI','meta_UCUI') and task!=pseudo:
                raise ValueError('Inner pseudo evaluation includes other task')
            if split in ('train','meta_UCSI') and task==pseudo:
                raise ValueError('Pseudo heldout task entered inner supervised train/source-city partition')
        partitions[split]=selection
    train_cities={str(inner_ds.df.iloc[int(i)].city) for i in inner_splits['train']}
    unseen_cities={str(inner_ds.df.iloc[int(i)].city)
                   for split in ('meta_UCSI','meta_UCUI') for i in inner_splits[split]}
    if train_cities & unseen_cities:
        raise ValueError('Inner city isolation failed')
    if not partitions['meta_SCUI'] or not partitions['meta_UCUI']:
        raise ValueError('Missing pseudo heldout evaluation partition')
    return partitions

def validated_vlm_jobs(manifest,records,model,max_pixels):
    from scripts.cache_ktrans_k7d import unique_jobs
    jobs=unique_jobs(manifest,model,max_pixels)
    valid={}
    for job in jobs:
        if job['condition'] not in ('image_task','image_only','text_only'):continue
        row=records.get(job['key'])
        if not row or row.get('parsed') is None:continue
        if any(row.get(field)!=job[field] for field in
               ('version','tile_id','task','condition','model','max_pixels','prompt')):
            continue
        valid[(job['tile_id'],job['task'],job['condition'])]={'key':job['key'],'parsed':row['parsed']}
    return valid

def visual_knowledge(tile,task,cache):
    """Return original VLM ordinal knowledge (scores + masks), no label access."""
    found=cache.get((tile,task,'image_task'))
    if not found:return None
    from urban_transfer.ktrans_k7d import encode_record
    v=encode_record(found['parsed'],'image_task').numpy()
    return {'knowledge_scores':v[:18].tolist(),'knowledge_masks':v[18:].tolist(),
            'knowledge_cache_key':found['key']}

def assemble_sample_rows(pseudo,split,rows,expert_outputs,labels,bank,cache):
    """Per sample/expert predictions converted using source-only semantic scale.

    The *same* pseudo-target semantic source moments scale every expert.
    Never use pseudo-target labels to calibrate these predictions.
    """
    if split not in ('meta_SCUI','meta_UCUI'):raise ValueError('Invalid inner eval split')
    z=np.asarray(expert_outputs,dtype=np.float64)
    y=np.asarray(labels,dtype=np.float64).reshape(-1)
    if z.shape!=(len(rows),len(bank.source_tasks)) or len(y)!=len(rows):
        raise ValueError('Expert outputs/labels not aligned to rows')
    if not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite expert outputs/labels')
    route=bank.routing[bank.task_to_all[pseudo]].detach().cpu().numpy().astype(float)
    mu=float(route@bank.source_means.detach().cpu().numpy())
    sigma=float(route@bank.source_stds.detach().cpu().numpy())
    if sigma<=0:raise ValueError('Nonpositive source-only scale')
    collected=[];coverage=Counter()
    for i,row in enumerate(rows):
        knowledge=visual_knowledge(row['tile_id'],pseudo,cache)
        if knowledge is None:
            coverage['missing_image_task_vlm']+=1
            continue
        coverage['with_vlm']+=1
        for j,expert in enumerate(bank.source_tasks):
            p=float(mu+sigma*z[i,j])
            err=p-float(y[i])
            collected.append({'pseudo_task':pseudo,'inner_partition':split,
                'city':row['city'],'tile_id':row['tile_id'],
                'image_path':row['image_path'],'outer_manifest_index':row['index'],
                'expert':expert,'expert_index':j,
                'label_log1p':float(y[i]),'expert_z':float(z[i,j]),
                'prediction_log1p':p,'error_log1p':err,'squared_error':err*err,
                'semantic_source_mean':mu,'semantic_source_std':sigma,
                'scale_source':'inner_source_only_semantic_moments',
                **knowledge})
    return collected,dict(coverage)

def audit_identity(rows,expected_outer):
    keys=set()
    for row in rows:
        if row['pseudo_task']==expected_outer or row['expert']==expected_outer:
            raise ValueError('Outer target leaked to competence dataset')
        if row['pseudo_task']==row['expert']:
            raise ValueError('Own-task expert in competence dataset')
        key=(row['pseudo_task'],row['inner_partition'],row['image_path'],row['expert'])
        if key in keys:raise ValueError('Duplicate competence record')
        keys.add(key)
    return len(keys)
