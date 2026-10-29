"""Latent routing tests: no download of Qwen weights required."""
import unittest
import torch
from urban_transfer.engine import Predictor, batch_latent_knowledge
from urban_transfer.model import ModelConfig

class FakeLatentProvider:
    def infer_hidden(self, *, image_path, city, task_description, source_tasks):
        return torch.arange(16, dtype=torch.float32) + float(len(task_description))

class TestLatentMode(unittest.TestCase):
    def test_provider_batch(self):
        batch={
          'city':['A','B'],
          'image_path':['one.png','two.png'],
          'task_description':['short','longer description'],
        }
        result=batch_latent_knowledge(batch, FakeLatentProvider(), ('POP','EDU'), torch.device('cpu'))
        self.assertEqual(tuple(result.shape),(2,16))
        self.assertFalse(torch.allclose(result[0],result[1]))
    def test_shared_train_eval_dispatch(self):
        from urban_transfer.engine import resolve_batch_knowledge
        from urban_transfer.knowledge import MockProvider
        batch={'city':['A','B'],'image_path':['one.png','two.png'],
               'task_description':['short','longer description']}
        latent=resolve_batch_knowledge(batch,FakeLatentProvider(),('POP','EDU'),torch.device('cpu'),'qwen3vl-latent')
        self.assertEqual(tuple(latent.shape),(2,16))
        explicit=resolve_batch_knowledge(batch,MockProvider(),('POP','EDU'),torch.device('cpu'),'mock')
        self.assertEqual(tuple(explicit.shape),(2,6))

    def test_projection_trains(self):
        cfg=ModelConfig(visual_dim=8, task_dim=6, hidden_dim=16, n_experts=4, top_k=2, n_sources=2)
        m=Predictor(cfg,latent_dim=16)
        batch={'image':torch.rand(2,3,32,32),'context':torch.zeros(2,4),'task':torch.rand(2,6)}
        pred=m(batch,torch.randn(2,16))
        self.assertEqual(tuple(pred.shape),(2,))
        pred.square().mean().backward()
        self.assertIsNotNone(m.latent_projection[1].weight.grad)
    def test_without_projection_original_path(self):
        cfg=ModelConfig(visual_dim=8,task_dim=6,hidden_dim=16,n_experts=4,top_k=2,n_sources=2)
        m=Predictor(cfg)
        batch={'image':torch.rand(2,3,32,32),'context':torch.zeros(2,4),'task':torch.rand(2,6)}
        self.assertEqual(tuple(m(batch,torch.rand(2,6)).shape),(2,))

if __name__=='__main__':unittest.main()
