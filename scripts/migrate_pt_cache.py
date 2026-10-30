"""Import legacy hashed .pt vector caches into the SQLite backend, without deletion."""
import argparse
from pathlib import Path
import re
import sqlite3
import numpy as np
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--source',required=True)
parser.add_argument('--destination',required=True)
args=parser.parse_args()
source=Path(args.source);destination=Path(args.destination)
destination.mkdir(parents=True,exist_ok=True)
dbpath=destination/'hidden_states.sqlite3'
total=0;invalid=0
with sqlite3.connect(str(dbpath),timeout=60) as db:
    db.execute('CREATE TABLE IF NOT EXISTS vectors (cache_key TEXT PRIMARY KEY, dim INTEGER NOT NULL, data BLOB NOT NULL)')
    for path in sorted(source.glob('*.pt')):
        key=path.stem
        if re.fullmatch(r'[0-9a-f]{64}',key) is None:
            continue
        try:
            tensor=torch.load(path,map_location='cpu',weights_only=True)
            if not isinstance(tensor,torch.Tensor) or tensor.ndim!=1 or not torch.isfinite(tensor).all():
                invalid+=1
                continue
            arr=tensor.detach().float().cpu().contiguous().numpy().astype('<f4',copy=False)
            db.execute('INSERT OR IGNORE INTO vectors(cache_key,dim,data) VALUES (?,?,?)',
                       (key,int(arr.size),sqlite3.Binary(arr.tobytes())))
            total+=1
            if total%1000==0:
                db.commit()
                print('Imported',total,flush=True)
        except (OSError,RuntimeError,ValueError,EOFError):
            invalid+=1
    db.commit()
    count=db.execute('select count(*) from vectors').fetchone()[0]
print(f'Complete: read {total} valid legacy files, skipped {invalid} invalid entries, {count} SQLite rows. Old files untouched.')
