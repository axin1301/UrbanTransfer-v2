import tempfile
import unittest
from pathlib import Path
import torch
from urban_transfer.task_profiles import task_texts
from urban_transfer.ktrans_text import cache_path,last_token_pool,qwen_task_vectors

class K4TextTests(unittest.TestCase):
    def test_profiles_are_complete_and_distinct(self):
        short=task_texts('short');structured=task_texts('structured')
        self.assertEqual(set(short),set(structured))
        self.assertEqual(len(short),10)
        self.assertNotEqual(short['digital_lte_intensity'],structured['digital_lte_intensity'])
    def test_profile_hash_changes_when_text_changes(self):
        a={'x':'one'};b={'x':'two'}
        self.assertNotEqual(cache_path('/tmp','model','structured',a),
                            cache_path('/tmp','model','structured',b))
    def test_pooling(self):
        hidden=torch.tensor([[[1.,0.],[2.,0.],[3.,0.]],[[0.,1.],[0.,2.],[0.,3.]]])
        mask=torch.tensor([[1,1,0],[1,1,1]])
        result=last_token_pool(hidden,mask)
        self.assertTrue(torch.equal(result,torch.tensor([[2.,0.],[0.,3.]])))
    def test_cache_only_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                qwen_task_vectors(root=d,profile='short',only_cached=True)
    def test_reviewed_json_requires_ten_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'profiles.json';p.write_text('{}')
            with self.assertRaises(ValueError):
                task_texts('llm',str(p))

if __name__=='__main__':unittest.main()
