"""Shared frozen RemoteCLIP -> tensor cache for K7 experiments.

One scan of source CSV and one SQLite read per unique image. Full-precision
CPU tensors are saved atomically and reused by all variants and seeds.
The cache is source-only, never includes official target-test CSVs.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.m3_features import FeatureStore
from urban_transfer.protocol import SOURCE_TASKS

VERSION = "k7-shared-tensors-v1"


def signature(source_csv, image_root, feature_dir, remoteclip_model, image_column):
    csv = Path(source_csv).resolve()
    db = Path(feature_dir).resolve() / "remoteclip_features.sqlite3"
    if not csv.is_file() or not db.is_file():
        raise FileNotFoundError(f"Source CSV or frozen RemoteCLIP DB absent: {csv}, {db}")
    stat_csv, stat_db = csv.stat(), db.stat()
    return {
        "version": VERSION, "csv": str(csv), "csv_size": stat_csv.st_size,
        "csv_mtime_ns": stat_csv.st_mtime_ns,
        "sqlite": str(db), "sqlite_size": stat_db.st_size,
        "sqlite_mtime_ns": stat_db.st_mtime_ns,
        "image_root": str(Path(image_root).resolve()),
        "encoder": "RemoteCLIP-" + remoteclip_model,
        "image_column": image_column,
    }


def build_tensor_cache(source_csv, image_root, feature_dir, model, output,
                       image_column=None):
    """Build once; never overwrite an existing mismatched cache in place."""
    output = Path(output)
    expected = signature(source_csv, image_root, feature_dir, model, image_column)
    if output.exists():
        stored = torch.load(output, map_location="cpu", weights_only=True)
        if stored.get("signature") == expected:
            print(f"[k7-cache] HIT {output} rows={len(stored['names'])} "
                  f"tiles={stored['unique_tiles']}", flush=True)
            return output
        raise ValueError("K7 tensor cache input signature changed; choose a NEW cache path")
    print("[k7-cache] scanning source CSV / resolving image paths...", flush=True)
    ds = ImageUrbanDataset(source_csv, image_root, partition="eval",
                           image_column=image_column)
    frame = ds.df
    if set(frame.indicator_column) != set(SOURCE_TASKS):
        raise ValueError("K7 source-only cache requires exactly six source tasks")
    if frame.city.astype(str).nunique() < 3:
        raise ValueError("K7 requires >= 3 source cities")
    store = FeatureStore(feature_dir, encoder_id="RemoteCLIP-" + model)
    # Source feature keys include tile path and its size + mtime; get() also
    # checks each row's stored feature, without decoding PNGs or loading CLIP.
    indices = {}
    features = []
    row_to_tile = []
    for row in tqdm(frame.itertuples(index=False), total=len(frame),
                    desc="K7 source tiles -> cached features", dynamic_ncols=True):
        path = str(ds.find_image(row))
        if path not in indices:
            vec = store.get(path)
            if vec is None:
                raise FileNotFoundError("Missing frozen RemoteCLIP tile: " + path)
            indices[path] = len(features)
            features.append(vec.float())
        row_to_tile.append(indices[path])
    visual = torch.stack(features)[torch.tensor(row_to_tile, dtype=torch.long)]
    context = torch.tensor(np.stack((
        frame.patch_lon.to_numpy(dtype=np.float32)/180.,
        frame.patch_lat.to_numpy(dtype=np.float32)/90.,
        np.log1p(np.maximum(frame.patch_area_km2.to_numpy(dtype=np.float32), 0.))/10.,
        np.ones(len(frame), dtype=np.float32)
    ), axis=1), dtype=torch.float32)
    labels = torch.tensor(frame.indicator_value_log1p.to_numpy(dtype=np.float32).copy())
    if not all(torch.isfinite(x).all() for x in (visual, context, labels)):
        raise ValueError("Nonfinite K7 input features")
    payload = {
        "signature": expected,
        "frame_csv": frame.to_csv(index=False),
        "x": visual, "context": context, "y": labels,
        "names": frame.indicator_column.astype(str).tolist(),
        "unique_tiles": len(features),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(output)
    print(f"[k7-cache] SAVED {output} rows={len(frame)} "
          f"unique_tiles={len(features)} visual_dim={visual.shape[1]}", flush=True)
    return output


def read_tensor_cache(path, source_csv, image_root, feature_dir, model,
                      image_column=None):
    stored = torch.load(path, map_location="cpu", weights_only=True)
    expected = signature(source_csv, image_root, feature_dir, model, image_column)
    if stored.get("signature") != expected:
        raise ValueError("Shared tensor cache no longer matches source CSV / DB")
    frame = pd.read_csv(io.StringIO(stored["frame_csv"]))
    names = stored["names"]
    if len(frame) != len(names) or len(frame) != len(stored["x"]):
        raise ValueError("Corrupt K7 shared tensor cache")
    if names != frame.indicator_column.astype(str).tolist():
        raise ValueError("Task identity mismatch in K7 tensor cache")
    return frame, stored["x"], stored["context"], stored["y"], names
