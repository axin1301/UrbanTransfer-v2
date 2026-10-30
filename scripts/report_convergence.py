"""Summarize M2.x run histories; no third-party plotting dependency."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('runs',nargs='+',help='training output directories')
a=p.parse_args()
for directory in a.runs:
    f=Path(directory)/'history.jsonl'
    if not f.exists():
        print(f'{directory}: history.jsonl missing');continue
    history=[json.loads(row) for row in f.read_text().splitlines() if row.strip()]
    if not history:
        print(f'{directory}: empty history');continue
    valid=[r for r in history if r.get('validation',{}).get('overall',{}).get('mse') is not None]
    if not valid:
        print(f'{directory}: no validation metrics');continue
    best=min(valid,key=lambda r:r['validation']['overall']['mse'])
    print(f'\n{directory}: {len(history)} epochs logged; best epoch={best["epoch"]}, val MSE={best["validation"]["overall"]["mse"]:.5f}')
    print(f'{"epoch":>6} {"train MSE":>13} {"val MSE":>13}')
    for r in history:
        tr=r['train']['mean_mse'];val=r['validation']['overall']['mse']
        print(f'{r["epoch"]:6d} {tr:13.5f} {val:13.5f}')
