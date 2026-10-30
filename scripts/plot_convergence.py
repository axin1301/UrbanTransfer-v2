"""Save a train/validation MSE plot to PNG. Requires matplotlib."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser()
p.add_argument('runs',nargs='+')
p.add_argument('--output',default='runs/m23_convergence.png')
a=p.parse_args()
fig,ax=plt.subplots(figsize=(9,5))
for name in a.runs:
    f=Path(name)/'history.jsonl'
    if not f.exists():continue
    records=[json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    if not records:continue
    epochs=[x['epoch'] for x in records]
    ax.plot(epochs,[x['train']['mean_mse'] for x in records],linestyle='--',label=Path(name).name+' train')
    ax.plot(epochs,[x['validation']['overall']['mse'] for x in records],label=Path(name).name+' val')
ax.set_xlabel('Epoch');ax.set_ylabel('MSE (log1p target)')
ax.legend(fontsize=8);ax.grid(alpha=.2)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
fig.tight_layout();fig.savefig(out,dpi=150)
print('saved',out)
