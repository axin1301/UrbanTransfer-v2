"""K6.2-A: audit K5-A nested folds before regional reliability supervision.

Read-only. No training, no fitting, no target-label access for decisions.
Checks source/pseudo-task/city/row provenance and frozen-feature dimensions.
Compatible tensor shape is NOT evidence that an outer regional router can
be transferred into a separately fitted inner expert representation.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_k4_profiles import load_model
from urban_transfer.protocol import SOURCE_TASKS


KEYS=('city','image_name','indicator_column')

def row_keys(frame):
    if not set(KEYS).issubset(frame.columns):
        raise ValueError(f'Missing row identity fields {set(KEYS)-set(frame.columns)}')
    return set(map(tuple,frame[list(KEYS)].itertuples(index=False,name=None)))

def validate_fold(outer_meta,outer_train,inner_meta,inner_df,inner_splits,
                  inner_model,pseudo,inner_ckpt):
    source=set(outer_meta['source_tasks'])
    if pseudo not in source:raise AssertionError('Pseudo target is not an outer source task')
    if inner_meta['heldout_task']!=pseudo:
        raise AssertionError('Nested checkpoint pseudo-target mismatch')
    if set(inner_model.source_tasks)!=source-{pseudo}:
        raise AssertionError('Inner experts include pseudo-target or miss source tasks')
    if outer_meta['heldout_task'] in inner_model.source_tasks:
        raise AssertionError('Outer heldout task present in inner expert bank')
    if set(inner_df.indicator_column)!=source:
        raise AssertionError('Nested CSV must contain all and only outer source tasks')
    # Exact identity and row counts guard against the nested checkpoint
    # inadvertently referring to a larger CSV or including other cities.
    if len(inner_df)!=len(outer_train) or row_keys(inner_df)!=row_keys(outer_train):
        raise AssertionError('Inner dataset does not exactly match outer TRAIN rows')
    splits={}
    for name in ('train','meta_SCUI','meta_UCSI','meta_UCUI'):
        ids=inner_splits[name]
        rows=inner_df.iloc[ids]
        splits[name]=rows
    if any(set(splits[a].index)&set(splits[b].index) for a,b in (
        ('train','meta_SCUI'),('train','meta_UCSI'),
        ('train','meta_UCUI'),('meta_SCUI','meta_UCSI'),
        ('meta_SCUI','meta_UCUI'),('meta_UCSI','meta_UCUI'))):
        raise AssertionError('Nested partition indices overlap')
    for name in ('train','meta_UCSI'):
        if pseudo in set(splits[name].indicator_column):
            raise AssertionError(f'Pseudo-target label leaked into {name}')
    for name in ('meta_SCUI','meta_UCUI'):
        if set(splits[name].indicator_column)!={pseudo}:
            raise AssertionError(f'Unexpected tasks in {name}')
    seen_cities=set(splits['train'].city)
    novel_cities=set(inner_meta['heldout_cities'])
    if seen_cities&novel_cities:
        raise AssertionError('Inner heldout cities overlap inner train cities')
    if not set(splits['meta_SCUI'].city).issubset(seen_cities):
        raise AssertionError('Inner SCUI contains unseen cities')
    if not set(splits['meta_UCUI'].city).issubset(novel_cities):
        raise AssertionError('Inner UCUI contains seen cities')
    for name in ('meta_UCSI','meta_UCUI'):
        if not set(splits[name].city).issubset(novel_cities):
            raise AssertionError(f'{name} does not use inner heldout cities')
    # The outer meta cities must never appear anywhere in the nested CSV.
    if set(inner_df.city)&set(outer_meta['heldout_cities']):
        raise AssertionError('Outer heldout cities entered nested dataset')
    shared=inner_model.shared
    return {
        'pseudo_target':pseudo,
        'checkpoint':str(inner_ckpt),
        'n_inner_source_experts':len(inner_model.source_tasks),
        'n_outer_train_rows':len(outer_train),
        'n_inner_train_rows':len(splits['train']),
        'n_inner_scui_rows':len(splits['meta_SCUI']),
        'n_inner_ucsi_rows':len(splits['meta_UCSI']),
        'n_inner_ucui_rows':len(splits['meta_UCUI']),
        'n_inner_train_cities':len(seen_cities),
        'n_inner_heldout_cities':len(novel_cities),
        'shared_output_dim':inner_model.shared[-2].out_features,
        'expert_input_dim':inner_model.experts[0][0].in_features,
        'pseudo_target_expert_excluded':True,
        'outer_heldout_cities_excluded':True,
        'row_provenance_exact':True,
        'regional_router_available':False,
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--k5a-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=p.parse_args()
    info=json.loads((args.k5a_dir/'protocol.json').read_text())
    outer_meta,outer_ds,outer_splits,outer_model,_=load_model(
        info['outer_checkpoint'],args.feature_dir,args.text_cache_dir,args.image_root)
    if info['outer_task']!=outer_meta['heldout_task']:
        raise AssertionError('Outer checkpoint target differs from K5-A protocol')
    if list(outer_model.source_tasks)!=list(info['inner_source_tasks']):
        raise AssertionError('K5-A source order differs from outer bank')
    if set(outer_model.source_tasks)!=set(SOURCE_TASKS)-{info['outer_task']}:
        raise AssertionError('Expected exactly five source experts')
    outer_train=outer_ds.df.iloc[outer_splits['train']]
    output=args.output or args.k5a_dir/'k62a_readiness'
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for pseudo in info['inner_source_tasks']:
        ckpt=args.k5a_dir/'inner_models'/pseudo/'last.pt'
        if not ckpt.is_file():raise FileNotFoundError(ckpt)
        inner_meta,inner_ds,inner_splits,inner_model,_=load_model(
            ckpt,args.feature_dir,args.text_cache_dir,args.image_root)
        row=validate_fold(outer_meta,outer_train,inner_meta,inner_ds.df,
                          inner_splits,inner_model,pseudo,ckpt)
        row['outer_shared_output_dim']=outer_model.shared[-2].out_features
        row['hidden_dimensions_match']=(
            row['shared_output_dim']==row['outer_shared_output_dim'])
        rows.append(row)
        print(f"{pseudo:<30} inner_train={row['n_inner_train_rows']:6d} "
              f"SCUI={row['n_inner_scui_rows']:5d} "
              f"UCUI={row['n_inner_ucui_rows']:5d} "
              f"experts={row['n_inner_source_experts']} "
              f"row_isolation=PASS",flush=True)
    with (output/'fold_audit.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    # Existing nested banks suffice for independently heldout pseudo-target
    # expert predictions, NOT for route-comparison reliability labels.
    report={
        'outer_heldout_task':info['outer_task'],
        'k5a_directory':str(args.k5a_dir),
        'source_folds_checked':len(rows),
        'expert_cross_task_holdout':'verified from nested metadata, rows, partitions',
        'independent_regional_router':'NOT AVAILABLE',
        'outer_K5B2_router_reusable':'NO: fitted against different outer expert bank',
        'tensor_shape_compatibility':'NOT sufficient to establish feature-space compatibility',
        'reliability_supervision_ready':False,
        'recommended_next_step':'Train each inner regional router on its own four-source inner training set without pseudo-target labels; evaluate on inner heldout pseudo-target cities.',
        'outer_checkpoint_caveat':'K4 best.pt previously selected on outer meta evaluation; DEV only.'
    }
    (output/'readiness.json').write_text(json.dumps(report,indent=2))
    print('K6.2-A readiness: nested expert holdouts VERIFIED; '
          'independent nested regional routers MISSING',flush=True)
    print('Saved',output/'fold_audit.csv',output/'readiness.json',flush=True)

if __name__=='__main__':main()
