"""All-outer-TRAIN, label-free VLM coverage planning and source manifest validation.

No labels or target-task images are inspected for job selection. This
manifest intentionally contains ONLY outer train rows, and is compatible
with the existing prompt-keyed K7-D cache script.
"""
import hashlib
from collections import Counter
from pathlib import Path

def full_source_manifest(ds,splits,meta,checkpoint):
    source=set(meta['source_tasks'])
    outer=meta['heldout_task']
    if outer in source:raise ValueError('Outer target included in source expert set')
    entries=[];seen=set();tiles={}
    for i in splits['train']:
        i=int(i)
        frame=ds.df.iloc[i]
        task=str(frame.indicator_column)
        city=str(frame.city)
        if task not in source or task==outer:
            raise ValueError('Non-source task in outer train')
        path=str(Path(ds.base.paths[i]).resolve())
        key=(path,task)
        if key in seen:raise ValueError('Duplicate image/task source sample')
        seen.add(key)
        tile=hashlib.sha256(path.encode()).hexdigest()[:20]
        if tile in tiles and tiles[tile]!=path:
            raise ValueError('SHA20 tile collision')
        tiles[tile]=path
        entries.append({'index':i,'partition':'train','tile_id':tile,
                        'image_path':path,'city':city,'task':task})
    if not entries or {r['task'] for r in entries}!=source:
        raise ValueError('Incomplete source train task coverage')
    if any(r['city'] in meta['heldout_cities'] for r in entries):
        raise ValueError('Outer heldout city included in train')
    entries.sort(key=lambda r:(r['task'],r['city'],r['image_path'],r['index']))
    return {'version':'k7d-manifest-v1',
      'source_manifest_mode':'outer-train-full-v1',
      'k4_checkpoint':str(Path(checkpoint).resolve()),
      'heldout_task':outer,
      'source_tasks':list(meta['source_tasks']),
      'seed':meta['seed'],
      'max_rows_per_partition':0,
      'rows':entries,'partition_counts':{'train':len(entries)},
      'note':'ALL outer TRAIN source rows only, no outer heldout task/city/label in manifest; K4 checkpoint still DEV-selected.'}

def validate_full_source_manifest(manifest,ds,splits,meta,checkpoint):
    if manifest.get('version')!='k7d-manifest-v1' or manifest.get('source_manifest_mode')!='outer-train-full-v1':
        raise ValueError('Expected all-source-only coverage manifest')
    if Path(manifest['k4_checkpoint']).resolve()!=Path(checkpoint).resolve():
        raise ValueError('Outer checkpoint provenance mismatch')
    if manifest['heldout_task']!=meta['heldout_task'] or list(manifest['source_tasks'])!=list(meta['source_tasks']):
        raise ValueError('Heldout/source expert mismatch')
    expected=full_source_manifest(ds,splits,meta,checkpoint)
    if manifest['rows']!=expected['rows'] or manifest.get('partition_counts')!=expected['partition_counts']:
        raise ValueError('All-source-only manifest does not exactly match complete outer train')
    return len(expected['rows'])

def valid_job(job,records):
    row=records.get(job['key'])
    return bool(row and row.get('parsed') is not None and all(
        row.get(k)==job[k] for k in
        ('version','tile_id','task','condition','model','max_pixels','prompt')))

def coverage_plan(manifest,records,model,max_pixels,old_train_keys=()):
    from scripts.cache_ktrans_k7d import unique_jobs
    if manifest.get('source_manifest_mode')!='outer-train-full-v1':
        raise ValueError('Refuse non-source-only manifest')
    jobs=unique_jobs(manifest,model,max_pixels)
    jobmap={(j['tile_id'],j['task'],j['condition']):j for j in jobs}
    pending=[j for j in jobs if not valid_job(j,records)]
    totals=Counter(j['condition'] for j in jobs)
    cached=Counter(j['condition'] for j in jobs if valid_job(j,records))
    count=Counter();old=set(old_train_keys)
    for row in manifest['rows']:
        task=row['task'];tile=row['tile_id']
        j=jobmap[(tile,task,'image_task')]
        count[(task,'total')]+=1
        if valid_job(j,records):count[(task,'cached_image_task')]+=1
        if (row['image_path'],task) in old:count[(task,'previous_manifest')]+=1
    task_rows=[{'task':task,'total_train_rows':count[(task,'total')],
                'in_previous_manifest':count[(task,'previous_manifest')],
                'cached_image_task':count[(task,'cached_image_task')],
                'pending_image_task':count[(task,'total')]-count[(task,'cached_image_task')]}
                for task in manifest['source_tasks']]
    return {'rows':len(manifest['rows']),
            'unique_tiles':len({r['tile_id'] for r in manifest['rows']}),
            'unique_image_task_pairs':len({(r['tile_id'],r['task']) for r in manifest['rows']}),
            'jobs_total':dict(totals),'jobs_cached_valid':dict(cached),
            'jobs_pending':dict(Counter(j['condition'] for j in pending)),
            'source_task_coverage':task_rows,
            'warning':'Counts are prompt-key exact. All outer train available as VLM input; inner pseudo-target task/city isolation still controls competence supervision.'},pending
