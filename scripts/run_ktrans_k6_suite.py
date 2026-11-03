"""Run and aggregate fixed K6 outer-LOTO experiments from an explicit manifest.

Does not silently skip folds or select winners using heldout labels.
Optional --train-missing performs source-only K5-A and K5-B2 training
from the specified K4 checkpoint, then evaluates all K6 variants.
"""
import argparse,csv,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from urban_transfer.protocol import SOURCE_TASKS
from scripts.run_ktrans_k6 import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path,
                   help='JSON mapping outer task to K4 checkpoint, K5-A directory, K5-B2 directory')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--train-missing',action='store_true')
    p.add_argument('--require-all-six',action='store_true')
    p.add_argument('--epochs-k5a',type=int,default=30)
    p.add_argument('--epochs-k5b2',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--eval-batch-size',type=int,default=1024)
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--image-root',default='/flash/project_462001391/xiyanxin/Urban_experience/output_files')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    folds=json.loads(args.manifest.read_text())
    if not isinstance(folds,dict) or not folds:raise ValueError('Manifest must be a nonempty mapping')
    if not set(folds).issubset(SOURCE_TASKS):raise ValueError('Only the six source tasks may be outer heldouts')
    if args.require_all_six and set(folds)!=set(SOURCE_TASKS):
        raise ValueError('A complete LOTO evaluation requires all six source tasks')
    all_rows=[]
    for task in SOURCE_TASKS:
        if task not in folds:continue
        item=folds[task]
        base=Path(item['k4_checkpoint'])
        a=Path(item['k5a_dir']);b=Path(item['k5b2_dir'])
        transfer=a/'protocol.json'
        composer=b/'composer.pt'
        if not base.is_file():raise FileNotFoundError(f'{task}: K4 checkpoint {base}')
        if args.train_missing:
            if not transfer.is_file():
                cmd=[sys.executable,'-u','scripts/run_ktrans_k5a.py',
                     '--outer-checkpoint',str(base),'--output',str(a),
                     '--epochs',str(args.epochs_k5a),'--batch-size',str(args.batch_size),
                     '--eval-batch-size',str(args.eval_batch_size),
                     '--feature-dir',args.feature_dir,'--text-cache-dir',args.text_cache_dir,
                     '--image-root',args.image_root]
                print('[k6-suite] training K5-A for',task,flush=True)
                subprocess.run(cmd,cwd=ROOT,check=True)
            if not composer.is_file():
                cmd=[sys.executable,'-u','scripts/run_ktrans_k5b.py',
                     '--checkpoint',str(base),'--output',str(b),'--stable-adapter',
                     '--epochs',str(args.epochs_k5b2),
                     '--batch-size',str(args.batch_size),'--eval-batch-size',
                     str(args.eval_batch_size),'--feature-dir',args.feature_dir,
                     '--text-cache-dir',args.text_cache_dir,'--image-root',args.image_root]
                print('[k6-suite] training K5-B2 for',task,flush=True)
                subprocess.run(cmd,cwd=ROOT,check=True)
        if not transfer.is_file() or not composer.is_file():
            raise FileNotFoundError(f'{task}: missing K5-A protocol or K5-B2 composer; '
                                    'use --train-missing to generate source-only components')
        fold_output=args.output/task
        fold_args=argparse.Namespace(composer_checkpoint=composer,k5a_protocol=transfer,
              output=fold_output,eval_batch_size=args.eval_batch_size,
              feature_dir=args.feature_dir,text_cache_dir=args.text_cache_dir,
              image_root=args.image_root)
        run(fold_args)
        with (fold_output/'evaluation.csv').open(newline='') as f:
            all_rows+=list(csv.DictReader(f))
        with (args.output/'aggregate.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(all_rows[0]))
            w.writeheader();w.writerows(all_rows)
    print('\nK6 fold coverage:',len(folds),'of 6 source tasks')
    if len(folds)<6:print('Partial development summary; do not label as six-fold aggregate.')
    for protocol in ('meta_SCUI','meta_UCSI','meta_UCUI'):
        print('\n'+protocol)
        for variant in ('k4','k5a','router_only','adapter_only','k5b2','k5c','k5d','k6_spatial','k6_scale_only','k6_full'):
            rows=[q for q in all_rows if q['variant']==variant and q['protocol']==protocol]
            if not rows:continue
            import numpy as np
            print(f'  {variant:<14} n_tasks={len(rows)} '
                  f'macro_MSE={np.mean([float(q["mse"]) for q in rows]):.3f} '
                  f'macro_nMSE={np.mean([1.-float(q["r2"]) for q in rows]):.3f} '
                  f'macro_R2={np.mean([float(q["r2"]) for q in rows]):.3f} '
                  f'macro_Pearson={np.mean([float(q["pearson"]) for q in rows]):.3f}')
    print('\nSaved:',args.output/'aggregate.csv')

if __name__=='__main__':main()
