"""M3 frozen RemoteCLIP feature store. One SQLite file; tile-only cache keys."""
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .data_m1 import ImageUrbanDataset

class FeatureStore:
    def __init__(self,root,encoder_id='RemoteCLIP-ViT-B-32'):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.encoder_id=encoder_id
        self.path=self.root/'remoteclip_features.sqlite3'
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL)')
    def connect(self):
        return sqlite3.connect(str(self.path),timeout=60)
    def key(self,path):
        p=Path(path).resolve();st=p.stat()
        return hashlib.sha256(json.dumps([self.encoder_id,str(p),st.st_size,st.st_mtime_ns]).encode()).hexdigest()
    def get(self,path):
        key=self.key(path)
        with self.connect() as db:
            result=db.execute('SELECT dim,vector FROM features WHERE key=?',(key,)).fetchone()
        if result is None:return None
        dim,blob=result
        if dim<1 or len(blob)!=2*dim:raise ValueError('Invalid feature cache entry')
        arr=np.frombuffer(blob,dtype='<f2').astype('float32')
        if not np.isfinite(arr).all():raise ValueError('Nonfinite cached embedding')
        return torch.from_numpy(arr)
    def put_many(self,items):
        rows=[]
        for path,v in items:
            a=v.detach().float().cpu().numpy()
            if a.ndim!=1 or not np.isfinite(a).all():raise ValueError('Invalid visual embedding')
            a=a.astype('<f2')
            rows.append((self.key(path),a.size,sqlite3.Binary(a.tobytes())))
        with self.connect() as db:
            db.executemany('INSERT OR REPLACE INTO features(key,dim,vector) VALUES (?,?,?)',rows)
    def count(self):
        with self.connect() as db:return db.execute('SELECT COUNT(*) FROM features').fetchone()[0]

class FrozenRemoteCLIP:
    def __init__(self,model_name='ViT-B-32',checkpoint=None,device='cuda'):
        try:
            import open_clip
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError('Install open-clip-torch and huggingface-hub for M3 precompute') from e
        if model_name not in ('ViT-B-32','ViT-L-14'):
            raise ValueError('Supported RemoteCLIP models: ViT-B-32, ViT-L-14')
        ckpt=checkpoint or hf_hub_download('chendelong/RemoteCLIP',f'RemoteCLIP-{model_name}.pt')
        self.model,_,self.preprocess=open_clip.create_model_and_transforms(model_name)
        weights=torch.load(ckpt,map_location='cpu',weights_only=True)
        self.model.load_state_dict(weights,strict=True)
        self.device=torch.device(device)
        self.model=self.model.to(self.device).eval()
        self.dim={'ViT-B-32':512,'ViT-L-14':768}[model_name]
    @torch.inference_mode()
    def encode(self,paths):
        inputs=[]
        for path in paths:
            with Image.open(path) as image:inputs.append(self.preprocess(image.convert('RGB')))
        image_batch=torch.stack(inputs).to(self.device)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            features=self.model.encode_image(image_batch)
        return torch.nn.functional.normalize(features.float(),dim=-1).cpu()

class CachedFeatureDataset(Dataset):
    """Do not decode PNG during MoE training/evaluation."""
    def __init__(self,csv_path,image_root,store,*,partition='eval',limit=None,task_dim=128,image_column=None,seed=42):
        self.base=ImageUrbanDataset(csv_path,image_root,partition=partition,limit=limit,
                                    task_dim=task_dim,image_column=image_column,seed=seed)
        self.df=self.base.df
        self.store=store
        self.paths=[str(self.base.find_image(row)) for row in self.df.itertuples(index=False)]
        # Check all unique images before starting a long training run.
        missing=[p for p in dict.fromkeys(self.paths) if self.store.get(p) is None]
        if missing:raise RuntimeError(f'{len(missing)} images lack frozen features; run precompute first. Example: {missing[0]}')
    def __len__(self):return len(self.base)
    def __getitem__(self,index):
        row=self.df.iloc[index]
        task=str(row.indicator_column)
        path=self.paths[index]
        ctx=torch.tensor([float(row.patch_lon)/180.,float(row.patch_lat)/90.,
            np.log1p(max(float(row.patch_area_km2),0.))/10.,1.],dtype=torch.float32)
        from .protocol import TASK_DESCRIPTIONS
        return {'image':self.store.get(path),'image_path':path,
                'context':ctx,'task':torch.from_numpy(self.base.task_lookup[task].copy()),
                'target':torch.tensor(float(row.indicator_value_log1p),dtype=torch.float32),
                'city':str(row.city),'task_name':task,'task_description':TASK_DESCRIPTIONS[task]}

def precompute(csv_paths,image_root,store,*,model_name='ViT-B-32',checkpoint=None,
               batch_size=64,image_column=None,limit=None,device='cuda'):
    # Deduplicate tiles across tasks and splits; never use CSV labels.
    paths={}
    for csv in csv_paths:
        ds=ImageUrbanDataset(csv,image_root,partition='eval',limit=limit,image_column=image_column)
        for row in ds.df.itertuples(index=False):
            p=str(ds.find_image(row));paths[p]=True
    missing=[p for p in paths if store.get(p) is None]
    print(f'[m3] unique images={len(paths)} missing={len(missing)} cached={len(paths)-len(missing)}',flush=True)
    if not missing:return
    encoder=FrozenRemoteCLIP(model_name,checkpoint,device=device)
    for start in range(0,len(missing),batch_size):
        chunk=missing[start:start+batch_size]
        features=encoder.encode(chunk)
        if features.shape[1]!=encoder.dim:raise RuntimeError('RemoteCLIP feature dimension mismatch')
        store.put_many(zip(chunk,features))
        if start//batch_size%10==0 or start+len(chunk)==len(missing):
            print(f'[m3] encoded {start+len(chunk)}/{len(missing)} new images',flush=True)
