# Single-file Qwen cache (SQLite)

The cache now writes frozen MLLM vectors to `<cache-dir>/hidden_states.sqlite3`, not one `.pt` file per tile. SQLite is indexed by the original image/model/city/task/source-indicator key, so repeated inference reuses the stored vector.

## Test

```bash
git pull origin main
python -m unittest discover -s tests -p 'test_sqlite_cache.py' -v
```

## Evaluate

Keep your previous command and specify a writable cache location, e.g.:

```bash
python -u run_m2.py evaluate --checkpoint runs/m23_qwen3vl-latent/best.pt \
  --scui-csv /path/to/scui.csv --image-root /path/to/images \
  --cache-dir /flash/project_462001391/xiyanxin/qwen_cache_sqlite \
  --output runs/m24_eval_qwen3vl-latent
```

The old `runs/qwen_cache/*.pt` files are NOT read automatically by the new SQLite backend.
Keep them until the new cache is tested. Existing entries can instead be migrated
with `python scripts/migrate_pt_cache.py --source runs/qwen_cache --destination runs/qwen_cache`.
The migration reads only cache entries with the exact 64-character hexadecimal key
format, and never touches checkpoint `.pt` files elsewhere.

Verify the SQLite entries before deleting old small cache files:

```bash
python - <<'PY'
import sqlite3
p='runs/qwen_cache/hidden_states.sqlite3'
with sqlite3.connect(p) as con:
    print('Cached vectors:',con.execute('select count(*) from vectors').fetchone()[0])
PY
```

Only **after** migration success and verification, if you explicitly want to clear the old
files, use `find runs/qwen_cache -maxdepth 1 -type f -name '*.pt' -delete`.
This permanently removes the old cache, not checkpoints elsewhere. Never delete
files until the SQLite database is verified and has backup if necessary.

For large full-protocol evaluation, one 2048-element FP32 vector is approximately 8 KiB
before SQLite overhead. The database may reach gigabytes, but consumes very few
inodes. Avoid concurrent jobs writing to the same database on a filesystem without
reliable SQLite file locking. Storing in local SSD scratch is preferable when supported.
