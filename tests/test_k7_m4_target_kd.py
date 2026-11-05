"""M4 source-only text teacher and KD gradients."""
import unittest
import torch

from urban_transfer.k7_target_distill import (
    teacher_source_distribution,distillation_loss)
from urban_transfer.k7_lora_composer import RelationLoRAComposer


class K7M4Tests(unittest.TestCase):
    def test_teacher_normalized_and_frozen(self):
        torch.manual_seed(42)
        src=torch.randn(2,16,requires_grad=True)
        target=torch.randn(16,requires_grad=True)
        edges=torch.randn(2,16,requires_grad=True)
        probs=teacher_source_distribution(src,target,edges)
        self.assertEqual(tuple(probs.shape),(2,))
        self.assertFalse(probs.requires_grad)
        self.assertAlmostEqual(probs.sum().item(),1.,places=6)
        self.assertTrue(bool((probs>=0.1).all()))
        self.assertTrue(bool((probs<=0.9).all()))

    def test_target_text_kd_updates_student_router(self):
        torch.manual_seed(42)
        model=RelationLoRAComposer(12,16,hidden=32,rank=4)
        x=torch.randn(5,12)
        ctx=torch.randn(5,4)
        src=torch.randn(2,16)
        target=torch.randn(16)
        edges=torch.randn(2,16)
        teacher=teacher_source_distribution(src,target,edges)
        _,mix,_=model(x,ctx,target.expand(5,-1),src,
                      edges.expand(5,-1,-1),torch.zeros(5,2),
                      return_aux=True)
        loss=distillation_loss(mix,teacher)
        self.assertGreaterEqual(loss.item(),0.)
        loss.backward()
        self.assertGreater(float(model.router.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.edge[1].weight.grad.abs().sum()),0.)

    def test_teacher_changes_for_distinct_relations(self):
        src=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        target=torch.tensor([1.,1.,0.])
        edges=torch.tensor([[1.,0.,0.],[0.,0.,1.]])
        a=teacher_source_distribution(src,target,edges)
        b=teacher_source_distribution(src,target,edges.flip(0))
        self.assertFalse(torch.allclose(a,b,atol=1e-5))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            teacher_source_distribution(torch.randn(3,4),torch.randn(4),
                                        torch.randn(2,4))
        with self.assertRaises(ValueError):
            distillation_loss(torch.randn(5,2),torch.tensor([.5,.5]))


if __name__=="__main__":
    unittest.main()
