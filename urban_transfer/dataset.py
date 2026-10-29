"""Data adapter for UrbanOpp CSV; NO target label information in model input."""
import ast
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class UrbanOppEmbeddingDataset(Dataset):
    def __init__(self, path, embedding_col='satellite_embedding_remoteclip', target_col='indicator_value_log1p', task_vectors=None):
        df=pd.read_parquet(path) if str(path).endswith('.parquet') else pd.read_csv(path)
        required=['patch_lon','patch_lat','patch_area_km2','indicator_column',target_col,embedding_col]
        missing=[x for x in required if x not in df]
        if missing: raise ValueError(f'Missing columns {missing}; supply a precomputed embedding dataset, not the labels-only CSV.')
        self.df=df.dropna(subset=[target_col]).reset_index(drop=True)
        if task_vectors is None: raise ValueError('Explicit task_vectors required; never infer task vectors from test labels')
        self.task_vectors=task_vectors
        self.embedding_col=embedding_col; self.target_col=target_col
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        vec=r[self.embedding_col]
        if isinstance(vec,str):
            try: vec=json.loads(vec)
            except json.JSONDecodeError: vec=ast.literal_eval(vec)
        image=torch.tensor(np.asarray(vec,dtype='float32'))
        ctx=torch.tensor([float(r.patch_lon)/180,float(r.patch_lat)/90,np.log1p(float(r.patch_area_km2))/10,1.0],dtype=torch.float32)
        task_name=r.indicator_column
        if task_name not in self.task_vectors: raise KeyError('Missing task description vector: '+task_name)
        return {'visual':image,'context':ctx,'task':torch.tensor(self.task_vectors[task_name],dtype=torch.float32),
                'target':torch.tensor(float(r[self.target_col]),dtype=torch.float32),
                'city':str(r.city),'task_name':str(task_name),'image_name':str(r.image_name)}
