import unittest
import torch
from urban_transfer import TransferModel,ModelConfig,MockProvider

class TestModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.cfg=ModelConfig(visual_dim=8,task_dim=6,hidden_dim=16,n_experts=4,top_k=2)
        self.x=torch.randn(3,8);self.c=torch.randn(3,4);self.t=torch.randn(3,6);self.k=torch.rand(3,9)
    def test_shape_backward(self):
        m=TransferModel(self.cfg); pred=m(self.x,self.c,self.t,self.k); self.assertEqual(tuple(pred.shape),(3,))
        pred.mean().backward();self.assertIsNotNone(m.knowledge_router.weight.grad)
    def test_zero_knowledge_init_equals_baseline(self):
        m=TransferModel(self.cfg); a=m(self.x,self.c,self.t,self.k)
        m.cfg.use_knowledge=False; b=m(self.x,self.c,self.t)
        self.assertTrue(torch.allclose(a,b))
    def test_knowledge_changes_routing_after_training(self):
        m=TransferModel(self.cfg)
        with torch.no_grad(): m.knowledge_router.weight.fill_(0.15)
        a=m(self.x,self.c,self.t,self.k,True)[1]['router_logits']
        b=m(self.x,self.c,self.t,1-self.k,True)[1]['router_logits']
        self.assertFalse(torch.allclose(a,b))
    def test_missing_knowledge(self):
        with self.assertRaises(ValueError): TransferModel(self.cfg)(self.x,self.c,self.t)
    def test_provider_validation(self):
        k=MockProvider().infer(image_path=None,city='x',task_description='y',source_tasks=('a','b'))
        self.assertEqual(len(k.validate(2).relevance),2)
if __name__=='__main__': unittest.main()
