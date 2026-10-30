"""SQLite-backed frozen Qwen latent cache: one DB, not one .pt per sample.

Stores detached float32 vectors, keyed by image identity + model + task prompt.
No prediction labels or trainable model features are written.
"""
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import numpy as np
import torch

class HiddenStateCache:
    def __init__(self, root, model_id, *, enabled=True):
        self.root=Path(root)
        self.model_id=model_id
        self.enabled=enabled
        self.db_path=self.root/'hidden_states.sqlite3'
        if enabled:
            self.root.mkdir(parents=True,exist_ok=True)
            with self._connect() as db:
                db.execute("""CREATE TABLE IF NOT EXISTS vectors (
                    cache_key TEXT PRIMARY KEY,
                    dim INTEGER NOT NULL,
                    data BLOB NOT NULL
                )""")

    def _connect(self):
        db=sqlite3.connect(str(self.db_path),timeout=60)
        db.execute('PRAGMA busy_timeout=60000')
        return db

    def _key(self, *, image_path, city, task_description, source_tasks):
        image=Path(image_path).resolve()
        stat=image.stat()
        record={'schema':1,'model_id':self.model_id,
                'image_path':str(image),'image_bytes':stat.st_size,
                'image_mtime_ns':stat.st_mtime_ns,'city':city,
                'task_description':task_description,
                'source_tasks':list(source_tasks)}
        return hashlib.sha256(
            json.dumps(record,sort_keys=True,ensure_ascii=False).encode()
        ).hexdigest()

    def get_or_compute(self, *, provider, image_path, city, task_description, source_tasks):
        if not self.enabled:
            return provider.infer_hidden(image_path=image_path,city=city,
                                         task_description=task_description,
                                         source_tasks=source_tasks)
        key=self._key(image_path=image_path,city=city,
                      task_description=task_description,source_tasks=source_tasks)
        try:
            with self._connect() as db:
                row=db.execute('SELECT dim,data FROM vectors WHERE cache_key=?',(key,)).fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError(f'Qwen SQLite cache read failed at {self.db_path}: {exc}') from exc
        if row is not None:
            dim,raw=row
            if dim > 0 and len(raw)==dim*4:
                value=np.frombuffer(raw,dtype='<f4').copy()
                if np.isfinite(value).all():
                    return torch.from_numpy(value)
        value=provider.infer_hidden(image_path=image_path,city=city,
                                    task_description=task_description,
                                    source_tasks=source_tasks)
        if not isinstance(value,torch.Tensor) or value.ndim!=1 or not torch.isfinite(value).all():
            raise ValueError('MLLM latent vector must be finite 1D Tensor')
        value=value.detach().float().cpu().contiguous()
        array=value.numpy().astype('<f4',copy=False)
        try:
            with self._connect() as db:
                db.execute('INSERT OR REPLACE INTO vectors(cache_key,dim,data) VALUES (?,?,?)',
                           (key,int(array.size),sqlite3.Binary(array.tobytes())))
        except sqlite3.Error as exc:
            raise RuntimeError(
                f'Qwen SQLite cache write failed at {self.db_path}: {exc}. '
                'Check directory permissions, free disk space and filesystem quota. '
                'Use --cache-dir to select a writable storage location.'
            ) from exc
        return value

    def __len__(self):
        if not self.enabled:return 0
        with self._connect() as db:
            return db.execute('SELECT COUNT(*) FROM vectors').fetchone()[0]
