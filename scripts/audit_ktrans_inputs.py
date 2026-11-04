"""Read-only K-TransMoE visual/context/text/label provenance audit.

No model training or label-dependent model selection. Source-only label
statistics; heldout rows are used only for input feature coverage.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.diagnose_k4_profiles import load_model
from urban_transfer.ktrans_text import qwen_task_vectors,cache_path
from urban_transfer.task_profiles import task_texts
from urban_transfer.protocol import TASK_DESCRIPTIONS

def cosine_matrix(vectors,names):
    arr=torch.stack([vectors[n].detach().float().cpu().flatten() for n in names])
    if not torch.isfinite(arr).all():
        raise ValueError('Nonfinite task vectors')
    if (arr.norm(dim=-1)<1e-10).any():
        raise ValueError('Near-zero task embedding')
    arr=torch.nn.functional.normalize(arr,dim=-1)
    return (arr@arr.T).numpy()

def summarize_inputs(context,images,labels=None):
    c=np.asarray(context,dtype=np.float64)
    v=np.asarray(images,dtype=np.float64)
    if c.ndim!=2 or c.shape[1]!=4 or v.ndim!=2 or len(c)!=len(v) or len(c)==0:
        raise ValueError('Invalid context or visual feature dimensions')
    if not np.isfinite(c).all() or not np.isfinite(v).all():
        raise ValueError('Nonfinite input')
    norms=np.linalg.norm(v,axis=-1)
    row={
        'n':len(c),'image_dim':v.shape[1],
        'image_norm_mean':float(norms.mean()),
        'image_norm_std':float(norms.std()),
        'unique_visual_rounded_6':int(len(np.unique(np.round(v,6),axis=0))),
        'image_axis_variance_mean':float(v.var(axis=0).mean()),
    }
    for j,name in enumerate(('longitude_180','latitude_90','log_area_10','intercept')):
        x=c[:,j]
        row.update({name+'_mean':float(x.mean()),
                    name+'_std':float(x.std()),
                    name+'_min':float(x.min()),
                    name+'_max':float(x.max())})
    if labels is not None:
        y=np.asarray(labels,dtype=np.float64)
        if len(y)!=len(c) or not np.isfinite(y).all():
            raise ValueError('Nonfinite/mismatched SOURCE labels')
        row.update({'source_label_mean':float(y.mean()),
                    'source_label_std':float(y.std()),
                    'source_label_min':float(y.min()),
                    'source_label_max':float(y.max())})
    return row

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-per-partition',type=int,default=3000)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=p.parse_args()
    if args.max_per_partition<1 or args.batch_size<1:p.error('Positive sample limits required')
    # load_model enforces exact checkpoint meta city split, source ordering
    # and cached-image availability for every row in the checkpoint CSV.
    meta,ds,splits,model,device=load_model(args.checkpoint,args.feature_dir,
                                           args.text_cache_dir,args.image_root)
    if meta.get('task_encoder')!='qwen3':
        raise ValueError('This audit targets frozen Qwen3 text-based K4')
    texts=task_texts(meta['task_profile'])
    vectors=qwen_task_vectors(root=args.text_cache_dir,
                model_id=meta['text_model_id'],profile=meta['task_profile'],
                only_cached=True)
    names=sorted(texts)
    if set(names)!=set(vectors) or set(names)!=set(TASK_DESCRIPTIONS):
        raise AssertionError('Task name/text/vector inventory mismatch')
    sim=cosine_matrix(vectors,names)
    cache=cache_path(args.text_cache_dir,meta['text_model_id'],meta['task_profile'],texts)
    if not cache.is_file():raise AssertionError('Versioned text cache not found')
    # Recompute semantic routing from source text and compare with K4 buffers.
    sources=list(model.source_tasks)
    temperature=float(meta['router_temperature'])
    idx={n:i for i,n in enumerate(names)}
    srcidx=[idx[n] for n in sources]
    expected=torch.softmax(torch.tensor(sim[:,srcidx])/temperature,dim=-1).numpy()
    actual=model.routing.detach().cpu().numpy()
    model_order=[idx[n] for n in model.task_names]
    if actual.shape!=expected.shape:
        raise AssertionError('Semantic routing matrix has unexpected shape')
    route_error=float(np.max(np.abs(actual-expected[model_order])))
    if route_error>2e-5:raise AssertionError(f'Task text route/cache mismatch: {route_error}')
    source_rows=ds.df.iloc[splits['train']]
    if set(source_rows.indicator_column)!=set(sources) or meta['heldout_task'] in set(source_rows.indicator_column):
        raise AssertionError('Train source-task isolation failed')
    if set(source_rows.city)&set(meta['heldout_cities']):
        raise AssertionError('Train vs meta heldout-city leakage')
    if not np.isfinite(source_rows.indicator_value_log1p.to_numpy(float)).all():
        raise AssertionError('Nonfinite source labels')
    if (source_rows.indicator_value_log1p.to_numpy(float)<-1e-6).any():
        raise AssertionError('Source log1p labels negative: inspect transform metadata')
    args.output.mkdir(parents=True,exist_ok=True)
    text_rows=[]
    for name in names:
        j=idx[name]
        pairs=sorted([(names[k],float(sim[j,k])) for k in range(len(names)) if k!=j],
                     key=lambda z:z[1],reverse=True)
        norm=float(vectors[name].float().norm())
        text_rows.append({'task':name,'text_length':len(texts[name]),
                 'embedding_dim':int(vectors[name].numel()),
                 'embedding_norm':norm,'nearest_task':pairs[0][0],
                 'nearest_similarity':pairs[0][1],
                 'source_route_top':sources[int(np.argmax(expected[j]))],
                 'source_route_top_weight':float(np.max(expected[j]))})
    with (args.output/'task_texts.json').open('w') as f:
        json.dump({'model_id':meta['text_model_id'],'profile':meta['task_profile'],
          'embedding_cache':str(cache),'tasks':texts},f,indent=2,ensure_ascii=False)
    with (args.output/'task_similarities.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['task']+names)
        for i,name in enumerate(names):w.writerow([name]+sim[i].tolist())
    with (args.output/'task_semantic_diagnostics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(text_rows[0]));w.writeheader();w.writerows(text_rows)

    from torch.utils.data import DataLoader,Subset
    from collections import Counter
    import pandas as pd
    input_rows=[];issues=[]
    # Deterministic per-partition sample with no access to heldout labels.
    for partition in ('train','meta_SCUI','meta_UCSI','meta_UCUI'):
        ids=splits[partition]
        chosen=ids[:args.max_per_partition]
        ctx=[];vis=[];image_paths=[];tasks=[];cities=[]
        for batch in DataLoader(Subset(ds,chosen),batch_size=args.batch_size,shuffle=False,num_workers=0):
            ctx.append(batch['context'].numpy())
            vis.append(batch['image'].numpy())
            image_paths.extend(batch['image_path'])
            tasks.extend(batch['task_name'])
            cities.extend(batch['city'])
        context=np.concatenate(ctx)
        image=np.concatenate(vis)
        # source labels audited only from outer train; never read labels
        # on meta-SCUI, -UCSI, or -UCUI for this input audit.
        labels=(ds.df.iloc[chosen].indicator_value_log1p.to_numpy(float)
                if partition=='train' else None)
        rec=summarize_inputs(context,image,labels)
        rec.update({'partition':partition,'sampled_rows':len(chosen),
                    'total_rows':len(ids),
                    'sampled_cities':len(set(cities)),
                    'sampled_tasks':len(set(tasks)),
                    'sampled_image_paths':len(set(image_paths))})
        input_rows.append(rec)
        for key in ('longitude_180','latitude_90','log_area_10'):
            if rec[key+'_std']<1e-6:
                issues.append(f'{partition}: constant geographic feature {key}')
        if rec['image_axis_variance_mean']<1e-10:
            issues.append(f'{partition}: almost constant cached visual features')
        if abs(rec['intercept_mean']-1.)>1e-6 or rec['intercept_std']>1e-6:
            issues.append(f'{partition}: expected constant intercept of 1')
        if rec['image_norm_mean']<.1 or rec['image_norm_mean']>10:
            issues.append(f'{partition}: surprising visual feature norm')
    # Identity-level duplicate rows within source training dataset.
    identities=list(map(tuple,ds.df[['city','image_name','indicator_column']].itertuples(index=False,name=None)))
    dup=len(identities)-len(set(identities))
    if dup:issues.append(f'{dup} repeated (city,image_name,indicator) row identities')
    if abs(float(np.diag(sim).min())-1.)>1e-4:
        issues.append('Text embedding self-similarity differs from 1')
    for row in text_rows:
        if row['nearest_similarity']>.98:
            issues.append(f"{row['task']}: unusually similar text vector to {row['nearest_task']} ({row['nearest_similarity']:.3f})")
    with (args.output/'input_partition_diagnostics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(input_rows[0]),extrasaction='ignore')
        w.writeheader();w.writerows(input_rows)
    report={'heldout_task':meta['heldout_task'],'k4_checkpoint':str(args.checkpoint),
       'source_training_rows':len(source_rows),'heldout_task_excluded_from_train':True,
       'heldout_cities_excluded_from_train':True,
       'text_cache_and_checkpoint_routing_consistent':True,
       'max_routing_consistency_error':route_error,
       'dataset_total_rows':len(ds),'duplicate_row_identities':dup,
       'sampling':'first max-per-partition rows; diagnostic only, not representative random sampling',
       'warnings':issues,'caveats':[
         'Text profile definitions are hypotheses and must be checked against data-generation methodology.',
         'Feature norms and distribution cannot establish aerial imagery semantic usefulness.',
         'Image file identity/cache coverage does not prove geospatial tile and label polygons match.',
         'Only outer train labels are summarized; heldout labels remain unused.',
         'K4 best.pt was previously selected via meta development validation.'
       ]}
    (args.output/'audit.json').write_text(json.dumps(report,indent=2))
    print(f'[audit] heldout={meta["heldout_task"]} text_dim={text_rows[0]["embedding_dim"]} '
          f'route_match_error={route_error:.8f} duplicate_row_ids={dup}',flush=True)
    for row in input_rows:
        print(f'[audit] {row["partition"]:<10} n={row["n"]} '
              f'image_norm={row["image_norm_mean"]:.3f} '
              f'vis_var={row["image_axis_variance_mean"]:.6f} '
              f'lon_std={row["longitude_180_std"]:.6f} '
              f'lat_std={row["latitude_90_std"]:.6f} '
              f'area_std={row["log_area_10_std"]:.6f}',flush=True)
    for row in text_rows:
        if row['task']==meta['heldout_task']:
            print(f'[audit] target_text nearest={row["nearest_task"]} '
                  f'cos={row["nearest_similarity"]:.3f} '
                  f'top_source={row["source_route_top"]} '
                  f'weight={row["source_route_top_weight"]:.3f}',flush=True)
    for msg in issues:print('[audit][CHECK]',msg,flush=True)
    print('[audit] saved to',args.output,flush=True)

if __name__=='__main__':main()
