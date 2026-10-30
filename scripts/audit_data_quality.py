"""Read-only UrbanOpp split/target audit; no train/test normalization or label editing."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from urban_transfer.data_paths import DEFAULT_SPLIT_DIR, SPLIT_FILES

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default=DEFAULT_SPLIT_DIR)
    p.add_argument('--output',default='runs/data_audit')
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    label='indicator_value_log1p'
    dfs={}
    summaries=[]
    problems=[]
    for split,filename in [('train',SPLIT_FILES['train_csv']),('SCUI',SPLIT_FILES['scui_csv']),
                            ('UCSI',SPLIT_FILES['ucsi_csv']),('UCUI',SPLIT_FILES['ucui_csv'])]:
        file=Path(args.data_dir)/filename
        df=pd.read_csv(file)
        needed=['city','image_name','indicator_column',label]
        absent=[c for c in needed if c not in df]
        if absent:raise ValueError(f'{file}: missing columns {absent}')
        dfs[split]=df
        y=pd.to_numeric(df[label],errors='coerce')
        bad=int((~np.isfinite(y.to_numpy(dtype=float))).sum())
        negative=int((y<0).sum())
        dup=int(df.duplicated(subset=['city','image_name','indicator_column']).sum())
        problems.append(dict(split=split,rows=len(df),missing_or_nonfinite_target=bad,negative_log1p=negative,
                             duplicate_city_tile_task=dup,cities=df.city.nunique(),tasks=df.indicator_column.nunique()))
        for task,g in df.assign(_target=y).groupby('indicator_column',dropna=False):
            z=g['_target'].dropna()
            z=z[np.isfinite(z)]
            summaries.append(dict(split=split,indicator=task,n=len(g),n_valid=len(z),
                mean=z.mean() if len(z) else np.nan,std=z.std(ddof=0) if len(z) else np.nan,
                minimum=z.min() if len(z) else np.nan,p01=z.quantile(.01) if len(z) else np.nan,
                median=z.median() if len(z) else np.nan,p99=z.quantile(.99) if len(z) else np.nan,
                maximum=z.max() if len(z) else np.nan,zero_fraction=(z==0).mean() if len(z) else np.nan))
    summary=pd.DataFrame(summaries)
    summary.to_csv(out/'indicator_distributions.csv',index=False,float_format='%.9g')
    pd.DataFrame(problems).to_csv(out/'quality_checks.csv',index=False)
    train=dfs['train']
    train_c=set(train.city.astype(str));train_t=set(train.indicator_column.astype(str))
    leakage=[]
    for split in ['SCUI','UCSI','UCUI']:
        df=dfs[split]
        overlap_c=len(train_c & set(df.city.astype(str)))
        overlap_t=len(train_t & set(df.indicator_column.astype(str)))
        leakage.append(dict(split=split,train_city_overlap=overlap_c,train_indicator_overlap=overlap_t,
                            unexpected_city_overlap=(split in ('UCSI','UCUI') and overlap_c>0),
                            unexpected_indicator_overlap=(split in ('SCUI','UCUI') and overlap_t>0)))
    pd.DataFrame(leakage).to_csv(out/'protocol_overlap.csv',index=False)
    # Explicitly do not compare a target-only indicator with nonexistent train stats.
    stats=summary[summary.split=='train'].set_index('indicator')
    comparisons=[]
    for _,r in summary[summary.split!='train'].iterrows():
        tr=stats.loc[r.indicator] if r.indicator in stats.index else None
        comparisons.append(dict(split=r.split,indicator=r.indicator,n=r.n,
           test_mean=r['mean'],test_std=r['std'],
           train_mean=tr['mean'] if tr is not None else np.nan,
           train_std=tr['std'] if tr is not None else np.nan,
           mean_shift_in_train_std=((r['mean']-tr['mean'])/tr['std']
             if tr is not None and pd.notna(tr['std']) and tr['std']>0 else np.nan),
           note='unseen indicator: no same-task training labels' if tr is None else 'same indicator available in train'))
    pd.DataFrame(comparisons).to_csv(out/'train_test_target_shift.csv',index=False,float_format='%.9g')
    print('QUALITY CHECKS')
    print(pd.DataFrame(problems).to_string(index=False))
    print('\nPROTOCOL OVERLAP (expected: UCSI/UCUI cities=0; SCUI/UCUI indicators=0)')
    print(pd.DataFrame(leakage).to_string(index=False))
    print('\nINDICATOR TARGET DISTRIBUTIONS (log1p units)')
    display=summary[['split','indicator','n','mean','std','median','p99','zero_fraction']]
    print(display.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print(f'\nSaved audit CSVs to {out}')
if __name__=='__main__':
    main()
