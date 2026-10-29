"""M2.2 controlled ablations, no model download required."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from PIL import Image
import pandas as pd
from urban_transfer.ablations import PermutedKnowledgeDataset
from urban_transfer.m2 import make_components, obtain_knowledge, train, evaluate

class TinyDataset:
    def __len__(self): return 5
    def __getitem__(self,i):
        return dict(image_path=f'image{i}.png',city=f'C{i}',task_description=f'task{i}',
                    image=torch.zeros(3,4,4),target=torch.tensor(float(i)))

def make_args(tmp,provider='no-llm'):
    return SimpleNamespace(
        train_csv=str(tmp/'train.csv'), scui_csv=str(tmp/'scui.csv'),
        ucsi_csv=None,ucui_csv=None,image_root=None,image_column=None,
        provider=provider,model_id='mock-qwen',output=str(tmp/'runs'),
        cache_dir=str(tmp/'cache'),no_cache=False,checkpoint=None,resume=None,
        epochs=1,batch_size=2,limit=None,seed=42,lr=1e-3,val_fraction=.25,
        task_dim=8,visual_dim=16,hidden_dim=16,experts=2,top_k=1,allow_missing_images=True)

class Tests(unittest.TestCase):
    def test_shuffle_preserves_supervised_examples(self):
        base=TinyDataset(); shuffled=PermutedKnowledgeDataset(base,seed=42)
        matches=0; donors=[]
        for i in range(len(base)):
            item=shuffled[i]
            self.assertEqual(item['image_path'],base[i]['image_path'])
            self.assertEqual(float(item['target']),float(i))
            self.assertNotEqual(item['knowledge_image_path'],item['image_path'])
            donors.append(item['knowledge_image_path'])
        self.assertEqual(len(set(donors)),len(base))
        again=PermutedKnowledgeDataset(base,seed=42)
        self.assertEqual(donors,[again[i]['knowledge_image_path'] for i in range(len(base))])

    def test_no_llm_does_not_load_provider(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,provider,model,device,cache=make_components(make_args(Path(d)),('food_accessibility',))
            self.assertIsNone(provider)
            self.assertFalse(cfg.use_knowledge)
            self.assertIsNone(model.latent_projection)
            self.assertIsNone(obtain_knowledge({},make_args(Path(d)),provider,(),device,cache))

    def test_no_llm_train_and_independent_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            tmp=Path(d)
            rows=[]
            for i in range(4):
                rows.append(dict(city='A',image_name=f'{i}.png',patch_lon=1.,patch_lat=2.,
                    patch_area_km2=1.,indicator_column='food_accessibility',
                    indicator_value_log1p=float(i),city_split='seen_city',
                    indicator_split='train_indicator'))
            pd.DataFrame(rows).to_csv(tmp/'train.csv',index=False)
            for row in rows:
                row['indicator_column']='healthcare_accessibility'
                row['indicator_split']='test_indicator'
            pd.DataFrame(rows).to_csv(tmp/'scui.csv',index=False)
            args=make_args(tmp)
            train(args)
            self.assertTrue((tmp/'runs'/'best.pt').exists())
            args.checkpoint=str(tmp/'runs'/'best.pt')
            results=evaluate(args)
            self.assertEqual(results['SCUI']['overall']['n'],4)

if __name__=='__main__': unittest.main()
