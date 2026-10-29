import tempfile
from pathlib import Path
import unittest
import torch
from PIL import Image
import pandas as pd
from urban_transfer.data_m1 import ImageUrbanDataset
from urban_transfer.engine import Predictor, batch_knowledge, metrics
from urban_transfer.knowledge import MockProvider
from urban_transfer.model import ModelConfig
from urban_transfer.protocol import SOURCE_TASKS, TARGET_TASKS, TASK_DESCRIPTIONS

def row(**kwargs):
    base=dict(city='A',image_name='a.png',patch_lon=0.,patch_lat=0.,patch_area_km2=3.,indicator_column='food_accessibility',indicator_value_log1p=.5,city_split='seen_city',indicator_split='train_indicator')
    base.update(kwargs);return base

class TestM1(unittest.TestCase):
    def test_real_schema_tasks(self):
        self.assertEqual(len(SOURCE_TASKS),6)
        self.assertEqual(len(TARGET_TASKS),4)
        self.assertTrue(set(SOURCE_TASKS+TARGET_TASKS).issubset(TASK_DESCRIPTIONS))
    def test_fail_missing_images(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'a.csv';pd.DataFrame([row(image_name='absent.png')]).to_csv(f,index=False)
            with self.assertRaises(FileNotFoundError): ImageUrbanDataset(f,d)[0]
    def test_explicit_absolute_image_column(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);image=root/'actual.png';Image.new('RGB',(16,16),(120,80,40)).save(image)
            f=root/'data.csv';pd.DataFrame([row(image_name='not_actual.png',image_path=str(image))]).to_csv(f,index=False)
            ds=ImageUrbanDataset(f,None,image_column='image_path')
            self.assertEqual(ds[0]['image_path'],str(image))
            self.assertEqual(tuple(ds[0]['image'].shape),(3,224,224))
    def test_relative_image_column(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'imgs').mkdir();Image.new('RGB',(8,8)).save(root/'imgs'/'x.png')
            f=root/'data.csv';pd.DataFrame([row(image_name='wrong.png',image_path='imgs/x.png')]).to_csv(f,index=False)
            ds=ImageUrbanDataset(f,None,image_column='image_path')
            self.assertEqual(ds[0]['image_path'],str((root/'imgs'/'x.png').resolve()))
    def test_target_label_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'a.csv';pd.DataFrame([row(indicator_column='healthcare_accessibility',indicator_split='test_indicator')]).to_csv(f,index=False)
            with self.assertRaises(ValueError): ImageUrbanDataset(f,d,partition='train')
    def test_batch_gradients(self):
        cfg=ModelConfig(n_sources=3);model=Predictor(cfg)
        batch={'image':torch.rand(2,3,224,224),'context':torch.zeros(2,4),'task':torch.rand(2,128),'target':torch.ones(2),
               'city':['A','B'],'image_path':['',''],'task_description':['foo','bar']}
        knowledge=batch_knowledge(batch,MockProvider(),('a','b','c'),torch.device('cpu'))
        y=model(batch,knowledge);torch.nn.functional.mse_loss(y,batch['target']).backward()
        self.assertIsNotNone(model.transfer.knowledge_router.weight.grad)
    def test_metrics(self):
        self.assertAlmostEqual(metrics([1,2,3],[1,2,3])['r2'],1.)

if __name__=='__main__':unittest.main()
