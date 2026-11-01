"""Resumable unattended K-TransMoE study: train, evaluate, aggregate.

Requires cached Qwen3-Embedding-8B short/structured task vectors beforehand.
Uses fixed seed/split/configuration; does NOT touch official external test labels.
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TASKS=('population_count','nighttime_light_intensity','education_accessibility',
       'food_accessibility','digital_modernization_tmi','digital_total_cell_intensity')
CONFIGS={
    'k2b1_remoteclip':('remoteclip','short',False),
    'k3a_remoteclip':('remoteclip','short',True),
    'k4_t1_qwen8b_short':('qwen3','short',True),
    'k4_t2_qwen8b_structured':('qwen3','structured',True),
}
ROUTES=[('semantic','shared',42),('uniform','shared',42)]
ROUTES += [('random','shared',s) for s in (41,42,43,44,45)]
ROUTES += [(spatial,scale,42) for spatial in ('semantic','uniform')
           for scale in ('semantic','uniform')]


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def run(cmd, logfile, deadline, state):
    logfile.parent.mkdir(parents=True,exist_ok=True)
    remaining=deadline-time.monotonic()
    if remaining<=60:
        raise TimeoutError('Study budget reached; resume later')
    print('[study]',stamp(),' '.join(cmd),flush=True)
    start=time.monotonic()
    with logfile.open('a') as out:
        out.write('\n===== '+stamp()+' '+ ' '.join(cmd)+' =====\n')
        out.flush()
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            code=proc.wait(timeout=max(1,remaining-30))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL)
            raise TimeoutError('Stopped a run at study budget, check checkpoint and resume')
    elapsed=round(time.monotonic()-start,2)
    state.append({'time':stamp(),'command':cmd,'log':str(logfile),'exit_code':code,'elapsed_s':elapsed})
    if code:
        raise RuntimeError(f'Exit code {code}: {logfile}')
    return True


def aggregate(root):
    all_rows=[]
    for condition in CONFIGS:
        for task in TASKS:
            for spatial,scale,seed in ROUTES:
                tag=f'{spatial}_scale_{scale}' if scale!='shared' else (
                    f'random_{seed}' if spatial=='random' else spatial)
                path=root/condition/'routing'/task/tag/'diagnostics.csv'
                if not path.exists():
                    continue
                with path.open(newline='') as f:
                    for row in csv.DictReader(f):
                        row.update({'condition':condition,'heldout':task,
                                    'spatial_route':spatial,'scale_route':scale,
                                    'route_seed':seed})
                        all_rows.append(row)
    if not all_rows:
        return 0
    dest=root/'aggregate';dest.mkdir(exist_ok=True)
    fields=list(all_rows[0])
    with (dest/'all_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(all_rows)
    summary=[]
    keys=sorted(set((r['condition'],r['spatial_route'],r['scale_route'],r['protocol'])
                    for r in all_rows))
    for condition,spatial,scale,protocol in keys:
        group=[r for r in all_rows if (r['condition'],r['spatial_route'],
                r['scale_route'],r['protocol'])==(condition,spatial,scale,protocol)]
        # Aggregate seeds within each fold first; report complete coverage explicitly.
        folds=[]
        for task in TASKS:
            selected=[r for r in group if r['heldout']==task]
            if not selected:continue
            av={}
            for metric in ('mse','r2','pearson','bias','pred_std','target_std'):
                values=[float(r[metric]) for r in selected if r.get(metric) not in ('',None,'None')]
                if values:av[metric]=sum(values)/len(values)
            folds.append(av)
        entry={'condition':condition,'spatial_route':spatial,'scale_route':scale,
               'protocol':protocol,'folds_present':len(folds),'folds_total':len(TASKS)}
        for metric in ('mse','r2','pearson','pred_std','target_std'):
            values=[f[metric] for f in folds if metric in f]
            entry[metric+'_macro']=sum(values)/len(values) if values else ''
        biases=[abs(f['bias']) for f in folds if 'bias' in f]
        entry['abs_bias_macro']=sum(biases)/len(biases) if biases else ''
        summary.append(entry)
    with (dest/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print('[study] summary:',dest/'summary.csv','rows:',len(all_rows),flush=True)
    return len(all_rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('runs/ktrans_overnight'))
    p.add_argument('--hours',type=float,default=7.75)
    p.add_argument('--limit',type=int,default=5000)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--eval-batch-size',type=int,default=256)
    p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
    p.add_argument('--feature-dir',default='runs/m3_features')
    p.add_argument('--only',choices=['all','train','evaluate','summarize'],default='all')
    p.add_argument('--conditions',nargs='+',choices=list(CONFIGS),default=list(CONFIGS))
    args=p.parse_args()
    if args.hours<=0:p.error('hours must be positive')
    root=(ROOT/args.root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+args.hours*3600
    state=[];failures=[]
    (root/'configuration.json').write_text(json.dumps({
        'conditions':args.conditions,'limit':args.limit,'epochs':args.epochs,
        'batch_size':args.batch_size,'eval_batch_size':args.eval_batch_size,
        'text_cache_dir':args.text_cache_dir,'feature_dir':args.feature_dir,
        'routes':ROUTES,'seed':42,'timestamp':stamp()},indent=2))
    try:
        if args.only in ('all','train'):
            for condition in args.conditions:
                encoder,profile,normalized=CONFIGS[condition]
                for task in TASKS:
                    directory=root/condition/task
                    ckpt=directory/'best.pt'
                    if ckpt.is_file():continue
                    cmd=[sys.executable,'-u','run_ktrans.py','train','--expert-bank',
                         *(['--normalized-experts'] if normalized else []),
                         '--task-encoder',encoder,'--task-profile',profile,
                         '--text-model-id','Qwen/Qwen3-Embedding-8B',
                         '--heldout-task',task,'--limit',str(args.limit),
                         '--meta-city-fraction','0.2','--batch-size',str(args.batch_size),
                         '--eval-batch-size',str(args.eval_batch_size),
                         '--epochs',str(args.epochs),'--patience','0','--seed','42',
                         '--feature-dir',args.feature_dir,'--text-cache-dir',args.text_cache_dir,
                         '--output',str(directory)]
                    try:run(cmd,root/'logs'/f'{condition}_{task}_train.log',deadline,state)
                    except TimeoutError:raise
                    except Exception as exc:
                        failures.append({'condition':condition,'task':task,'stage':'train','error':str(exc)})
                        print('[study] failed, continuing:',exc,flush=True)
        if args.only in ('all','evaluate'):
            for condition in args.conditions:
                _,_,normalized=CONFIGS[condition]
                for task in TASKS:
                    ckpt=root/condition/task/'best.pt'
                    if not ckpt.is_file():continue
                    for spatial,scale,seed in ROUTES:
                        if scale!='shared' and not normalized:continue
                        tag=f'{spatial}_scale_{scale}' if scale!='shared' else (
                            f'random_{seed}' if spatial=='random' else spatial)
                        dest=root/condition/'routing'/task/tag
                        if (dest/'diagnostics.csv').is_file():continue
                        cmd=[sys.executable,'-u','scripts/analyze_ktrans_predictions.py',
                             '--checkpoint',str(ckpt),'--routing-mode',spatial,
                             '--scale-routing-mode',scale,'--routing-seed',str(seed),
                             '--batch-size',str(args.eval_batch_size),
                             '--feature-dir',args.feature_dir,
                             '--text-cache-dir',args.text_cache_dir,'--output',str(dest)]
                        try:run(cmd,root/'logs'/f'{condition}_{task}_{tag}.log',deadline,state)
                        except TimeoutError:raise
                        except Exception as exc:
                            failures.append({'condition':condition,'task':task,'stage':tag,'error':str(exc)})
                            print('[study] failed, continuing:',exc,flush=True)
    except TimeoutError as exc:
        print('[study] budget reached:',exc,flush=True)
    finally:
        (root/'job_history.json').write_text(json.dumps(state,indent=2))
        (root/'failures.json').write_text(json.dumps(failures,indent=2))
        count=aggregate(root)
        print('[study] finished/paused',stamp(),'records=',count,'failures=',len(failures),flush=True)


if __name__=='__main__':main()
