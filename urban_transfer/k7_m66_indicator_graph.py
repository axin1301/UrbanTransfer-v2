"""M6.6 label-free physical-neighbor graph, no outcome/indicator access.

Returns row-aligned [N,K] physical neighbor row indices, distances in km
(log1p / 10) and validity mask. Neighbors are restricted to the same city,
self tiles excluded and duplicate indicator rows deduplicated.
Only source-CSV unlabeled visual/coordinate context is used.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch


def physical_neighbor_graph(frame,k=8):
    from scipy.spatial import cKDTree
    if k<1 or not {"city","image_name","patch_lon","patch_lat"}.issubset(frame):
        raise ValueError("Invalid label-free neighbor graph inputs")
    n=len(frame)
    city=frame.city.astype(str).to_numpy()
    tile=frame.image_name.astype(str).to_numpy()
    lon=frame.patch_lon.to_numpy(dtype=np.float64)
    lat=frame.patch_lat.to_numpy(dtype=np.float64)
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError("Nonfinite tile coordinates")
    index=np.zeros((n,k),dtype=np.int64)
    distance=np.zeros((n,k),dtype=np.float32)
    mask=np.zeros((n,k),dtype=np.bool_)
    for c in sorted(set(city)):
        rows=np.flatnonzero(city==c)
        physical={}
        for i in rows:
            physical.setdefault(tile[int(i)],[]).append(int(i))
        keys=sorted(physical)
        representative=np.asarray([physical[key][0] for key in keys],
                                  dtype=np.int64)
        positions=np.stack([lon[representative],lat[representative]],axis=-1)
        for key,rep in zip(keys,representative):
            ix=np.asarray(physical[key],dtype=np.int64)
            if not np.allclose(lon[ix],lon[rep],rtol=0,atol=1e-7) or not np.allclose(
                lat[ix],lat[rep],rtol=0,atol=1e-7):
                raise ValueError("Duplicate physical tile has inconsistent coordinates")
        lat0=float(np.deg2rad(positions[:,1].mean()))
        xy=np.stack((positions[:,0]*111.32*np.cos(lat0),
                     positions[:,1]*110.574),axis=-1)
        tree=cKDTree(xy)
        count=min(k+1,len(keys))
        dd,ii=tree.query(xy,k=count)
        dd=np.asarray(dd).reshape(len(keys),count)
        ii=np.asarray(ii).reshape(len(keys),count)
        for row_number,key in enumerate(keys):
            good=ii[row_number]!=row_number
            neighbors=ii[row_number,good][:k]
            distances=dd[row_number,good][:k]
            output_rows=np.asarray(physical[key],dtype=np.int64)
            index[output_rows,:]=int(representative[row_number])
            count_here=len(neighbors)
            if count_here:
                index[output_rows,:count_here]=representative[neighbors]
                distance[output_rows,:count_here]=np.log1p(
                    distances).astype(np.float32)/10.
                mask[output_rows,:count_here]=True
    return (torch.from_numpy(index),torch.from_numpy(distance),
            torch.from_numpy(mask))


def cached_physical_neighbor_graph(frame,manifest_path,path,k=8):
    manifest_path=Path(manifest_path).resolve()
    path=Path(path)
    digest=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    meta={"schema":"physical-neighbor-graph-m66-v1",
          "manifest_sha256":digest,"n_rows":len(frame),"k":k}
    if path.is_file():
        saved=torch.load(path,map_location="cpu",weights_only=True)
        if saved["metadata"]!=meta:
            raise ValueError("M6.6 neighbor index cache mismatch")
        index,distance,mask=(saved[key]
                             for key in ("index","distance","mask"))
        if (index.shape!=(len(frame),k) or distance.shape!=index.shape or
            mask.shape!=index.shape or index.min()<0 or index.max()>=len(frame)):
            raise ValueError("Corrupt M6.6 graph cache")
        print(f"[m66] neighbor graph HIT {path}",flush=True)
        return index,distance,mask
    result=physical_neighbor_graph(frame,k=k)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    torch.save({"metadata":meta,
                "index":result[0],"distance":result[1],"mask":result[2]},temp)
    temp.replace(path)
    print(f"[m66] neighbor graph SAVED {path}",flush=True)
    return result
