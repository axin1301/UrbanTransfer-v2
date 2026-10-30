import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from urban_transfer.m3_features import FeatureStore,CachedFeatureDataset
from urban_transfer.m3 import M3Predictor
from urban_transfer.model import ModelConfig

class M3Tests(unittest.TestCase):
    def test_unique_tile_store_and_cached_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            images=root/'Adana'/'arcgis-imagery-FUA'
            images.mkdir(parents=True)
            image=images/'tile.png'
            Image.fromarray(np.zeros((16,16,3),dtype=np.uint8)).save(image)
            csv=root/'test.csv'
            pd.DataFrame([{'city':'Adana','image_name':'tile.png',
              'indicator_column':t,'indicator_value_log1p':1.2,
              'patch_lon':35.,'patch_lat':37.,'patch_area_km2':1.0}
              for t in ['population_count','food_accessibility']]).to_csv(csv,index=False)
            store=FeatureStore(root/'features')
            store.put_many([(str(image),torch.randn(512))])
            self.assertEqual(store.count(),1)
            ds=CachedFeatureDataset(csv,root,store)
            self.assertEqual(len(ds),2)
            self.assertEqual(ds[0]['image'].shape,(512,))
            self.assertTrue(torch.equal(ds[0]['image'],ds[1]['image']))
            cfg=ModelConfig(visual_dim=768,n_sources=2,use_knowledge=False)
            model=M3Predictor(cfg,512)
            a=ds[0];b={key:(val.unsqueeze(0) if isinstance(val,torch.Tensor) else [val])
                           for key,val in a.items()}
            y=model(b,None)
            self.assertEqual(y.shape,(1,))
            self.assertTrue(torch.isfinite(y).all())

    def test_invalidate_modified_image(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            img=root/'img.png';img.write_bytes(b'first')
            store=FeatureStore(root/'features')
            store.put_many([(str(img),torch.ones(512))])
            self.assertIsNotNone(store.get(str(img)))
            img.write_bytes(b'second with different byte count')
            self.assertIsNone(store.get(str(img)))

if __name__=='__main__':unittest.main()
