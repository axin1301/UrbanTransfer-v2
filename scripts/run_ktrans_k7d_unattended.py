"""One-command on-server K7-D cache->source-only ablations pipeline.

This process runs only when the USER launches it on the MI250X node.
It never submits a remote job from ChatGPT. All child commands and
stderr are written to a local log; failures stop the pipeline.
"""
import argparse,csv,json,subprocess,sys,time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.cache_ktrans_k7d import merge_records,unique_jobs

ROOT=Path(__file__).resolve().parents[1]

def run_command(cmd,log):
    print('[k7d-auto] RUN', ' '.join(map(str,cmd)),flush=True)
    t0=time.monotonic()
    with log.open('a') as stream:
        stream.write('\nRUN: '+' '.join(map(str,cmd))+'\n')
        stream.flush()
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in proc.stdout:
            print(line,end='',flush=True)
            stream.write(line)
            stream.flush()
        code=proc.wait()
        stream.write(f'EXIT {code} elapsed_s={time.monotonic()-t0:.1f}\n')
    if code!=0:
        raise RuntimeError(f'Pipeline command failed with exit={code}: {cmd}')
    return code

def cache_quality(manifest,cache,reuse,model,max_pixels,condition=None):
    jobs=unique_jobs(json.loads(Path(manifest).read_text()),model,max_pixels)
    if condition:jobs=[j for j in jobs if j['condition']==condition]
    records=merge_records([reuse,cache])
    missing=[j for j in jobs if j['key'] not in records or records[j['key']].get('parsed') is None]
    return len(jobs),len(missing),[(j['tile_id'],j['task'],j['condition']) for j in missing[:5]]

def aggregate(paths,out):
    data=[]
    for seed,path in paths:
        if not path.is_file():raise FileNotFoundError(path)
        with path.open() as f:
            for row in csv.DictReader(f):
                row['seed']=seed
                data.append(row)
    if not data:raise ValueError('No evaluation records')
    bykey=defaultdict(list)
    base={}
    for r in data:
        key=(r['seed'],r['outer_task'],r['protocol'])
        if r['variant']=='k5a':base[key]=float(r['mse'])
    for row in data:
        key=(row['variant'],row['outer_task'],row['protocol'])
        reference=base[(row['seed'],row['outer_task'],row['protocol'])]
        bykey[key].append((float(row['mse']),float(row['pearson']),
                           float(row['bias']),float(row['mse'])-reference))
    import statistics
    rows=[]
    for (variant,task,protocol),values in sorted(bykey.items()):
        rows.append({'variant':variant,'outer_task':task,'protocol':protocol,
           'seeds':len(values),'mse_mean':statistics.mean(v[0] for v in values),
           'mse_std':statistics.pstdev(v[0] for v in values),
           'pearson_mean':statistics.mean(v[1] for v in values),
           'bias_mean':statistics.mean(v[2] for v in values),
           'delta_mse_vs_k5a_mean':statistics.mean(v[3] for v in values)})
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    for row in rows:
        print(f"[k7d-auto] {row['protocol']:<10} {row['variant']:<12} "
              f"MSE={row['mse_mean']:.3f} (+/-{row['mse_std']:.3f}) "
              f"delta_vs_K5A={row['delta_mse_vs_k5a_mean']:+.3f}",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--k5a-protocol',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--reuse-cache',type=Path,default=Path('runs/k7b_pilot/evidence.jsonl'))
    p.add_argument('--batch-size',type=int,default=8,help='8B VLM inference batch size')
    p.add_argument('--train-batch-size',type=int,default=256)
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--max-new-tokens',type=int,default=440)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--preflight',type=int,default=8,
                   help='Number of image-only images to test before large inference')
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    args=p.parse_args()
    seeds=[int(x) for x in args.seeds.split(',') if x.strip()]
    if not seeds or len(seeds)!=len(set(seeds)):p.error('Seeds must be nonempty and unique')
    for file in (args.checkpoint,args.k5a_protocol,args.manifest):
        if not file.is_file():p.error('Missing '+str(file))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    cache=args.output_dir/'evidence.jsonl'
    log=args.output_dir/'pipeline.log'
    common=[sys.executable,'-u','scripts/cache_ktrans_k7d.py',
           '--manifest',str(args.manifest),
           '--output',str(cache),
           '--reuse-cache',str(args.reuse_cache),
           '--model',args.model,
           '--max-pixels',str(args.max_pixels),
           '--max-new-tokens',str(args.max_new_tokens),
           '--batch-size',str(args.batch_size)]
    status=args.output_dir/'pipeline_status.json'
    state={'stage':'starting','status':'RUNNING','seeds':seeds,
           'checkpoint':str(args.checkpoint),'manifest':str(args.manifest),
           'note':'DEV ONLY: K4 source trunk not independently inner crossfitted'}
    def save(stage,status_value='RUNNING',error=None):
        state.update(stage=stage,status=status_value)
        if error:state['error']=error
        status.write_text(json.dumps(state,indent=2))
    try:
        save('image_only_preflight')
        allcount,missing,_=cache_quality(args.manifest,cache,args.reuse_cache,args.model,
                                         args.max_pixels,'image_only')
        if missing and args.preflight:
            run_command(common+['--conditions','image_only',
                                '--max-new-jobs',str(args.preflight)],log)
            total,remaining,samples=cache_quality(args.manifest,cache,args.reuse_cache,
                args.model,args.max_pixels,'image_only')
            # The preflight jobs are deterministic first N uncached image-only identities.
            if remaining>max(0,missing-args.preflight):
                raise RuntimeError('Image-only v2 preflight INVALID. '
                                   'Inspect evidence.jsonl and pipeline.log before full run.')
            print(f'[k7d-auto] preflight passed; image_only valid={total-remaining}/{total}',flush=True)
        save('full_vlm_cache')
        run_command(common,log)
        total,missing,samples=cache_quality(args.manifest,cache,args.reuse_cache,
                                            args.model,args.max_pixels)
        if missing:raise RuntimeError(f'Incomplete VLM cache {total-missing}/{total}; '
                                      f'examples={samples}; no training started')
        print(f'[k7d-auto] cache coverage {total}/{total} complete',flush=True)
        output_paths=[]
        for seed in seeds:
            target=args.output_dir/f'seed_{seed}'
            eval_file=target/'evaluation.csv'
            save(f'training_seed_{seed}')
            if eval_file.is_file():
                print(f'[k7d-auto] seed={seed} found evaluation, skip training',flush=True)
            else:
                cmd=[sys.executable,'-u','scripts/run_ktrans_k7d.py',
                     '--checkpoint',str(args.checkpoint),
                     '--k5a-protocol',str(args.k5a_protocol),
                     '--manifest',str(args.manifest),
                     '--knowledge-cache',str(cache),
                     '--reuse-cache',str(args.reuse_cache),
                     '--output',str(target),
                     '--model',args.model,
                     '--max-pixels',str(args.max_pixels),
                     '--epochs',str(args.epochs),
                     '--batch-size',str(args.train_batch_size),
                     '--seed',str(seed)]
                run_command(cmd,log)
            output_paths.append((seed,eval_file))
        save('aggregation')
        aggregate(output_paths,args.output_dir/'summary.csv')
        save('done','COMPLETED')
        print('[k7d-auto] COMPLETED summary=',args.output_dir/'summary.csv',flush=True)
    except BaseException as exc:
        save('failed','FAILED',str(exc))
        print('[k7d-auto] FAILED:',exc,flush=True)
        raise

if __name__=='__main__':main()
