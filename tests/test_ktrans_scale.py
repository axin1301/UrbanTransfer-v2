import unittest
import torch
from urban_transfer.model import ModelConfig
from urban_transfer.ktrans_scale import TaskScalePredictor

class K2ATests(unittest.TestCase):
    def test_shape_gradients_and_finite_outputs(self):
        torch.manual_seed(42)
        cfg=ModelConfig(visual_dim=32,task_dim=16,hidden_dim=16,
                        n_experts=3,top_k=2,n_sources=5,use_knowledge=False)
        m=TaskScalePredictor(cfg,input_dim=8,task_dim=16)
        batch={'image':torch.randn(4,8),'context':torch.randn(4,4),
               'task':torch.randn(4,16),'target':torch.randn(4)}
        y=m(batch,None)
        self.assertEqual(tuple(y.shape),(4,))
        self.assertTrue(torch.isfinite(y).all())
        (y-batch['target']).square().mean().backward()
        self.assertIsNotNone(m.scale_head[-1].weight.grad)
        self.assertGreater(m.scale_head[-1].weight.grad.abs().sum().item(),0)
    def test_initial_identity(self):
        cfg=ModelConfig(visual_dim=32,task_dim=16,hidden_dim=16,
                        n_experts=3,top_k=2,n_sources=5,use_knowledge=False)
        m=TaskScalePredictor(cfg,input_dim=8,task_dim=16)
        batch={'image':torch.randn(3,8),'context':torch.randn(3,4),'task':torch.randn(3,16)}
        with torch.no_grad():
            self.assertTrue(torch.allclose(m(batch,None),m.backbone(batch,None),atol=1e-6))

if __name__=='__main__':unittest.main()
