import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
import pandas as pd
from urban_transfer.cache import HiddenStateCache
from urban_transfer.checkpoints import save_checkpoint,load_checkpoint
from urban_transfer.m2 import train,evaluate
from urban_transfer.model import ModelConfig
from urban_transfer.engine import Predictor

class CountingProvider:
    def __init__(self):
        self.calls=0
    def infer_hidden(self,*,image_path,city,task_description,source_tasks):
        self.calls+=1
        return torch.arange(8,dtype=torch.float32)

def args(train_csv,scui_csv,output,**overrides):
    kwargs=dict(train_csv=str(train_csv),scui_csv=str(scui_csv),ucsi_csv=None,ucui_csv=None,
        image_root=None,image_column=None,provider='mock',model_id='mock-model',
        output=str(output),cache_dir=str(Path(output)/'cache'),no_cache=False,
        checkpoint=None,resume=None,epochs=1,batch_size=2,limit=None,seed=42,lr=1e-3,
        val_fraction=0.25,task_dim=8,visual_dim=16,hidden_dim=16,experts=2,
        top_k=1,allow_missing_images=True)
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)

def make_csv(tmp,filename,task,indicator_split,city_split):
    rows=[]
    for i in range(4):
        rows.append(dict(city='A',image_name=f'{i}.png',patch_lon=float(i),patch_lat=0.,
            patch_area_km2=1.,indicator_column=task,indicator_value_log1p=float(i)/4,
            city_split=city_split,indicator_split=indicator_split))
    path=tmp/filename;pd.DataFrame(rows).to_csv(path,index=False)
    return path

class TestM2(unittest.TestCase):
    def test_cache_key_and_hit(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);image=p/'image.png';image.write_bytes(b'fake-key-only')
            provider=CountingProvider();cache=HiddenStateCache(p/'cache','model-version')
            kw=dict(image_path=str(image),city='A',task_description='Task A',source_tasks=('pop',))
            a=cache.get_or_compute(provider=provider,**kw)
            b=cache.get_or_compute(provider=provider,**kw)
            self.assertEqual(provider.calls,1)
            self.assertTrue(torch.equal(a,b))
            cache.get_or_compute(provider=provider,**(kw|{'task_description':'Task B'}))
            self.assertEqual(provider.calls,2)

    def test_checkpoint_roundtrip_resume(self):
        with tempfile.TemporaryDirectory() as td:
            cfg=ModelConfig(visual_dim=16,task_dim=8,hidden_dim=16,n_experts=2,top_k=1,n_sources=1)
            m=Predictor(cfg);optim=torch.optim.AdamW(m.parameters(),lr=1e-3)
            f=Path(td)/'model.pt'
            save_checkpoint(f,model=m,optimizer=optim,cfg=cfg,meta={'source_tasks':['x']},epoch=1)
            loaded=load_checkpoint(f,model=m,optimizer=optim,restore_rng=True)
            self.assertEqual(loaded['epoch'],1)

    def test_independent_mock_train_evaluate(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            train_csv=make_csv(p,'train.csv','food_accessibility','train_indicator','seen_city')
            scui_csv=make_csv(p,'scui.csv','healthcare_accessibility','test_indicator','seen_city')
            config=args(train_csv,scui_csv,p/'run')
            train(config)
            self.assertTrue((p/'run'/'best.pt').exists())
            config.checkpoint=str(p/'run'/'best.pt')
            evaluate(config)
            self.assertTrue((p/'run'/'metrics.json').exists())
            config.resume=str(p/'run'/'last.pt')
            config.epochs=2
            train(config)
            self.assertEqual(load_checkpoint(p/'run'/'last.pt',model=Predictor(ModelConfig(
                visual_dim=16,task_dim=8,hidden_dim=16,n_experts=2,top_k=1,n_sources=1)))['epoch'],2)

if __name__=='__main__':
    unittest.main()
