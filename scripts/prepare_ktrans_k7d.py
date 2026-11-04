"""K7-D stage 1: freeze a deterministic, source-isolated row/visual manifest.

Manifest records never contain indicator labels; K4 exact split indices
and cached RemoteCLIP image paths are reused. All four meta partitions
are sampled without looking at indicator values or dev predictions.
"""
import argparse,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.diagnose_k4_profiles import load_model

PARTITIONS=('train','meta_SCUI','meta_UCSI','meta_UCUI')

def choose(df,indices,limit,seed):
    """City/task round-robin sample by IDs, independent of label magnitudes."""
    groups=defaultdict(list)
    for idx in indices:
        row=df.iloc[idx]
        groups[(str(row.city),str(row.indicator_column))].append(int(idx))
    rng=np.random.default_rng(seed)
    for group in groups.values():rng.shuffle(group)
    keys=sorted(groups);rng.shuffle(keys)
    if limit==0:limit=len(indices)
    chosen=[]
    while len(chosen)<min(limit,len(indices)):
        did=False
        for key in keys:
            if groups[key]:
                chosen.append(groups[key].pop())
                did=True
                if len(chosen)>=limit:break
        if not did:break
    return chosen

def make_manifest(ds,splits,meta,k4,limit,seed):
    paths=ds.base.paths
    if len(paths)!=len(ds.df):
        raise AssertionError('Feature paths not aligned to sampled dataset')
    samples=[]
    for partition in PARTITIONS:
        ids=choose(ds.df,splits[partition],limit,seed)
        for index in ids:
            row=ds.df.iloc[index]
            path=str(Path(paths[index]).resolve())
            tile=hashlib.sha256(path.encode()).hexdigest()[:20]
            samples.append({'index':int(index),'partition':partition,
               'tile_id':tile,'image_path':path,'city':str(row.city),
               'task':str(row.indicator_column)})
    tasks={p:set() for p in PARTITIONS}
    cities={p:set() for p in PARTITIONS}
    for row in samples:
        tasks[row['partition']].add(row['task'])
        cities[row['partition']].add(row['city'])
    if tasks['train']!={str(t) for t in meta['source_tasks']}:
        raise ValueError('Insufficient source-task coverage in sampled train partition')
    heldout=meta['heldout_task']
    if heldout in tasks['train'] or tasks['meta_SCUI']!={heldout} or tasks['meta_UCUI']!={heldout}:
        raise AssertionError('Outer task split contamination')
    if cities['train'] & cities['meta_UCSI'] or cities['train'] & cities['meta_UCUI']:
        raise AssertionError('Outer heldout city leaked to K7-D training')
    return {'version':'k7d-manifest-v1','k4_checkpoint':str(Path(k4).resolve()),
            'heldout_task':heldout,'source_tasks':list(meta['source_tasks']),
            'seed':seed,'max_rows_per_partition':limit,'rows':samples,
            'partition_counts':{p:sum(row['partition']==p for row in samples) for p in PARTITIONS},
            'note':'No indicator labels in manifest; K4 checkpoint is meta-dev-selected.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--max-rows-per-partition',type=int,default=512,
                   help='0=all rows; otherwise deterministic city×task balanced subset')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=p.parse_args()
    if args.max_rows_per_partition<0:p.error('Negative sample limit')
    meta,ds,splits,_,_=load_model(args.checkpoint,args.feature_dir,args.text_cache_dir,args.image_root)
    manifest=make_manifest(ds,splits,meta,args.checkpoint,args.max_rows_per_partition,args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(manifest,indent=2))
    print('[k7d] task=',manifest['heldout_task'],'counts=',manifest['partition_counts'],
          'unique_tile_task=',len({(r['tile_id'],r['task']) for r in manifest['rows']}),flush=True)
    print('[k7d] saved',args.output,flush=True)

if __name__=='__main__':main()
