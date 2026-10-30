import sqlite3
import tempfile
import unittest
from pathlib import Path
import torch
from urban_transfer.cache import HiddenStateCache

class FakeProvider:
    def __init__(self): self.calls=0
    def infer_hidden(self,*,image_path,city,task_description,source_tasks):
        self.calls+=1
        return torch.tensor([0.25,0.5,0.75],dtype=torch.float32)

class SQLiteCacheTests(unittest.TestCase):
    def test_cache_hit_single_database_and_task_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            image=root/'tile.png'; image.write_bytes(b'fake-image-for-key-test')
            cache=HiddenStateCache(root/'cache','qwen-test')
            p=FakeProvider()
            kw=dict(image_path=str(image),city='Adana',
                    task_description='healthcare',source_tasks=('population',))
            first=cache.get_or_compute(provider=p,**kw)
            second=cache.get_or_compute(provider=p,**kw)
            self.assertEqual(p.calls,1)
            self.assertTrue(torch.equal(first,second))
            self.assertEqual(len(cache),1)
            cache.get_or_compute(provider=p,**(kw|{'task_description':'education'}))
            self.assertEqual(p.calls,2)
            self.assertEqual(len(cache),2)
            self.assertEqual(len(list((root/'cache').glob('*.pt'))),0)
            self.assertTrue((root/'cache'/'hidden_states.sqlite3').is_file())
            with sqlite3.connect(root/'cache'/'hidden_states.sqlite3') as db:
                self.assertEqual(db.execute('select count(*) from vectors').fetchone()[0],2)
            # Cache survives another process/instance; no stale in-memory dependency.
            again=HiddenStateCache(root/'cache','qwen-test')
            again.get_or_compute(provider=p,**kw)
            self.assertEqual(p.calls,2)

    def test_batch_reuses_cache_and_deduplicates_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            image=root/'tile.png';image.write_bytes(b'fake')
            cache=HiddenStateCache(root/'cache','qwen-test')
            p=FakeProvider()
            req=dict(image_path=str(image),city='A',task_description='task',source_tasks=('x',))
            other=dict(req,task_description='another')
            result=cache.get_or_compute_many(provider=p,requests=[req,req,other])
            self.assertEqual(p.calls,2)
            self.assertEqual(len(result),3)
            self.assertTrue(torch.equal(result[0],result[1]))
            again=cache.get_or_compute_many(provider=p,requests=[other,req])
            self.assertEqual(p.calls,2)
            self.assertEqual(len(again),2)

    def test_cache_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'tile.png';image.write_bytes(b'fake')
            cache=HiddenStateCache(root/'cache','qwen-test',enabled=False)
            p=FakeProvider()
            kw=dict(image_path=str(image),city='A',task_description='task',source_tasks=('x',))
            cache.get_or_compute(provider=p,**kw)
            cache.get_or_compute(provider=p,**kw)
            self.assertEqual(p.calls,2)
            self.assertFalse((root/'cache').exists())

if __name__=='__main__': unittest.main()
