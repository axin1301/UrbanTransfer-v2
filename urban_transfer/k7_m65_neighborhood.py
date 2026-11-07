"""M6.5 label-free per-city physical-tile neighborhood representation.

K=8 geographic nearest physical tiles within the same city, with no
indicator values, indicator IDs, or test labels used as neighborhood
features. Same tile under different source indicator rows is deduped.
Nearby context is transductive within the available *source CSV*
imagery: all partitions' satellite embeddings can be considered as
unlabeled spatial context, never as targets or supervised examples.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def city_neighborhood_features(frame,visual,k=8):
    from scipy.spatial import cKDTree
    if k<1 or visual.ndim!=2 or len(frame)!=len(visual):
        raise ValueError("Invalid neighborhood tensor inputs")
    n,d=visual.shape
    if not {"city","image_name","patch_lon","patch_lat"}.issubset(frame):
        raise ValueError("Physical tile coordinates missing")
    # No target value or target indicator is read here.
    city=frame.city.astype(str).to_numpy()
    image=frame.image_name.astype(str).to_numpy()
    lon=frame.patch_lon.to_numpy(dtype=np.float64)
    lat=frame.patch_lat.to_numpy(dtype=np.float64)
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError("Nonfinite geography")
    features=visual.detach().cpu().float().numpy()
    result=np.zeros((n,3*d+2),dtype=np.float32)
    for city_name in sorted(set(city)):
        rows=np.flatnonzero(city==city_name)
        physical={}
        for idx in rows:
            physical.setdefault(image[int(idx)],[]).append(int(idx))
        tiles=sorted(physical)
        tile_rows=[physical[name] for name in tiles]
        coords=np.asarray(
            [[lon[indices[0]],lat[indices[0]]] for indices in tile_rows],
            dtype=np.float64)
        vectors=np.stack([features[indices[0]] for indices in tile_rows])
        # Same physical image should have consistent coordinates/features
        # regardless of Population/Food/Education row duplication.
        for ids,position,vec in zip(tile_rows,coords,vectors):
            if (not np.allclose(np.stack([lon[ids],lat[ids]],axis=-1),
                                position,atol=1e-7,rtol=0)
                or not np.allclose(features[ids],vec,atol=1e-5,rtol=0)):
                raise ValueError("Inconsistent repeated physical tile")
        lat0=float(np.deg2rad(np.mean(coords[:,1])))
        xy=np.stack([coords[:,0]*111.32*np.cos(lat0),
                     coords[:,1]*110.574],axis=1)
        tree=cKDTree(xy)
        neighbors=min(k+1,len(tiles))
        distances,positions=tree.query(xy,k=neighbors)
        distances=np.asarray(distances).reshape(len(tiles),neighbors)
        positions=np.asarray(positions).reshape(len(tiles),neighbors)
        for j,ids in enumerate(tile_rows):
            good=positions[j]!=j
            nn=positions[j,good][:k]
            dist=distances[j,good][:k]
            if len(nn):
                neigh=vectors[nn]
                mean=neigh.mean(axis=0)
                spread=neigh.std(axis=0)
                average_distance=float(np.mean(dist))
                count=float(len(nn))/k
            else:
                mean=vectors[j].copy()
                spread=np.zeros(d,dtype=np.float32)
                average_distance=0.
                count=0.
            # neighborhood mean, std and image-minus-neighbor contrast
            context=np.concatenate([
                mean,spread,vectors[j]-mean,
                np.asarray([np.log1p(average_distance)/10.,count],
                           dtype=np.float32)])
            result[ids]=context
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite neighborhood features")
    return torch.from_numpy(result)


def load_or_build_neighborhood(frame,visual,manifest_path,output,k=8):
    manifest_path=Path(manifest_path).resolve()
    output=Path(output)
    manifest_bytes=manifest_path.read_bytes()
    digest=hashlib.sha256(manifest_bytes).hexdigest()
    meta={"manifest_sha256":digest,"k":k,
          "n_rows":len(frame),"visual_dim":visual.shape[-1],
          "schema":"city-neighborhood-v1"}
    if output.exists():
        cached=torch.load(output,map_location="cpu",weights_only=True)
        if cached.get("metadata")!=meta:
            raise ValueError("Neighborhood cache metadata mismatch")
        features=cached["context"]
        if features.shape!=(len(frame),3*visual.shape[-1]+2):
            raise ValueError("Corrupt neighborhood cache")
        print(f"[k7-m65] neighborhood cache HIT {output}",flush=True)
        return features
    print(f"[k7-m65] building label-free neighborhood k={k}...",flush=True)
    features=city_neighborhood_features(frame,visual,k=k)
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_name(output.name+".tmp")
    torch.save({"metadata":meta,"context":features},temp)
    temp.replace(output)
    print(f"[k7-m65] neighborhood SAVED {output} shape={tuple(features.shape)}",flush=True)
    return features
