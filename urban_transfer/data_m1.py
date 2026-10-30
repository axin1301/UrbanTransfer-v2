"""Image-backed UrbanOpp rows and strict split safeguards."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def stratified_sample(df, limit, seed=42):
    """Deterministic balanced sample across all available city × indicator strata."""
    if limit is None or limit >= len(df):
        return df.reset_index(drop=True)
    if limit < 1:
        raise ValueError('limit must be positive')
    keys = ['city', 'indicator_column']
    # Seeded shuffle within each group, then round-robin across groups.
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(indices).tolist() for _, indices in
              df.groupby(keys, sort=True).indices.items()]
    if len(groups) > limit:
        raise ValueError(f'limit={limit} is smaller than {len(groups)} city × indicator strata; increase --limit')
    chosen = []
    while len(chosen) < limit:
        progressed = False
        for group in groups:
            if group and len(chosen) < limit:
                chosen.append(group.pop())
                progressed = True
        if not progressed:
            break
    return df.iloc[chosen].reset_index(drop=True)
import torch
from torch.utils.data import Dataset
from PIL import Image
from .protocol import TASK_DESCRIPTIONS

def task_vector(text, dim=128):
    """Deterministic string baseline only; replace with pretrained text embeddings for publication."""
    words=text.lower().replace('-', ' ').split()
    arr=np.zeros(dim, dtype=np.float32)
    for word in words:
        digest=hashlib.sha256(word.encode()).digest()
        i=int.from_bytes(digest[:4], 'big') % dim
        arr[i]+=1 if digest[4]%2 else -1
    return arr/(np.linalg.norm(arr)+1e-8)

class ImageUrbanDataset(Dataset):
    def __init__(self, csv_path, image_root, *, partition='train', limit=None, task_dim=128,
                 require_images=True, allow_target_train=False, image_column=None, seed=42):
        self.df=pd.read_csv(csv_path)
        needed={'city','image_name','indicator_column','patch_lon','patch_lat','patch_area_km2','indicator_value_log1p'}
        if not needed.issubset(self.df.columns): raise ValueError(f'Missing columns: {needed-set(self.df.columns)}')
        if partition == 'train':
            if 'city_split' in self.df and not self.df.city_split.eq('seen_city').all():
                raise ValueError('Training data contains unseen cities')
            if 'indicator_split' in self.df and not self.df.indicator_split.eq('train_indicator').all():
                if not allow_target_train: raise ValueError('Training data contains held-out indicators')
        self.df=self.df.dropna(subset=['indicator_value_log1p']).reset_index(drop=True)
        if limit is not None:
            if partition == 'train':
                self.df=stratified_sample(self.df,limit,seed)
            else:
                self.df=self.df.iloc[:limit].reset_index(drop=True)
        self.root=Path(image_root) if image_root else None
        self.csv_dir=Path(csv_path).resolve().parent
        self.image_column=image_column
        if self.image_column and self.image_column not in self.df.columns:
            raise ValueError(f'Image path column {self.image_column!r} not found in {csv_path}')
        if not self.image_column and self.root is None and require_images:
            raise ValueError('Supply --image-root or --image-column (or both)')
        self.task_dim=task_dim
        self.require_images=require_images
        self.task_lookup={t:task_vector(s,task_dim) for t,s in TASK_DESCRIPTIONS.items()}
        missing=set(self.df.indicator_column.unique())-set(TASK_DESCRIPTIONS)
        if missing: raise ValueError(f'Unknown indicator IDs: {missing}. Update protocol.py with the exact schema.')
    def __len__(self): return len(self.df)
    def find_image(self,r):
        if self.image_column:
            raw=r[self.image_column]
            if pd.notna(raw) and str(raw).strip():
                path=Path(str(raw).strip()).expanduser()
                candidates=[path] if path.is_absolute() else ([self.root/path] if self.root else [])+[self.csv_dir/path]
                for candidate in candidates:
                    if candidate.is_file(): return candidate.resolve()
                raise FileNotFoundError(f'Explicit image path {raw!r} not found; checked: {candidates}')
            raise ValueError(f'Empty image path for row city={r.city}, task={r.indicator_column}')
        filename=Path(str(r.image_name))
        candidates=[self.root/filename]
        city_names=[str(r.city)]
        if 'city_query' in r and pd.notna(r.city_query):
            city_names.append(str(r.city_query))
        for city_name in dict.fromkeys(city_names):
            candidates.extend([self.root/city_name/filename,
                               self.root/city_name/'arcgis-imagery-FUA'/filename])
        for candidate in candidates:
            if candidate.is_file():return candidate.resolve()
        raise FileNotFoundError(f'Image not found for city={r.city}, image={filename}; checked {candidates}')
    def __getitem__(self,i):
        r=self.df.iloc[i]
        path=self.find_image(r) if self.require_images else ''
        if path:
            with Image.open(path) as im:
                arr=np.asarray(im.convert('RGB').resize((224,224)),dtype=np.float32).copy()/255.
            image=torch.from_numpy(arr).permute(2,0,1)
        else:
            image=torch.zeros(3,224,224)
        task=str(r.indicator_column)
        ctx=torch.tensor([float(r.patch_lon)/180.,float(r.patch_lat)/90.,np.log1p(max(float(r.patch_area_km2),0.))/10.,1.],dtype=torch.float32)
        return {'image':image,'image_path':str(path),'context':ctx,'task':torch.from_numpy(self.task_lookup[task].copy()),
                'target':torch.tensor(float(r.indicator_value_log1p),dtype=torch.float32),
                'city':str(r.city),'task_name':task,'task_description':TASK_DESCRIPTIONS[task]}
