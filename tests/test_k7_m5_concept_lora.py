"""K7 M5: concept grounded source LoRA tests (CPU, no external checkpoints)."""
import unittest
import torch

from urban_transfer.k7_concept_lora import (
    ConceptLoRAComposer,rank_targets,concept_supervision,rank_distillation)
from urban_transfer.k7_m5_concepts import visual_concept_targets,CONCEPTS


class K7M5Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=ConceptLoRAComposer(12,16,hidden=32,rank=8)
        self.visual=torch.randn(7,12)
        self.context=torch.randn(7,4)
        self.target=torch.randn(7,16)
        self.sources=torch.randn(2,16)
        self.edges=torch.randn(7,2,16)
        self.flags=torch.zeros(7,2)

    def test_frozen_remoteclip_concepts_source_calibration(self):
        anchor=torch.nn.functional.normalize(torch.randn(4,12),dim=-1)
        train=torch.randn(7,12)
        result=visual_concept_targets(train,train,anchor)
        self.assertEqual(tuple(result.shape),(7,4))
        self.assertTrue(bool((result>0).all() and (result<1).all()))
        self.assertEqual(len(CONCEPTS),4)

    def test_m5_rank_alignment_and_target_kd_gradients(self):
        out,mix,gates,activations=self.model(
            self.visual,self.context,self.target,self.sources,
            self.edges,self.flags,return_concepts=True)
        self.assertEqual(tuple(out.shape),(7,))
        self.assertEqual(tuple(mix.shape),(7,2))
        self.assertEqual(tuple(gates.shape),(7,2,8))
        self.assertTrue(torch.allclose(mix,torch.full_like(mix,.5)))
        teacher=rank_targets(torch.randn(2,16),torch.randn(4,16),rank=8)
        self.assertTrue(torch.allclose(teacher.sum(-1),torch.ones(2)))
        aligned=concept_supervision(
            activations,torch.sigmoid(torch.randn(7,4)))
        kd=rank_distillation(gates,teacher,rank=8)
        loss=out.square().mean()+aligned+kd
        loss.backward()
        self.assertGreater(float(self.model.source_a.grad.abs().sum()),0.)
        self.assertGreater(float(self.model.source_b.grad.abs().sum()),0.)
        self.assertGreater(float(self.model.rank_gate.weight.grad.abs().sum()),0.)

    def test_text_swap_changes_prediction_after_lora_training(self):
        with torch.no_grad():
            self.model.source_b.normal_(0,.1)
            self.model.rank_gate.weight.normal_(0,.01)
        a=self.model(self.visual,self.context,self.target,self.sources,
                     self.edges,self.flags)
        b=self.model(self.visual,self.context,self.target,self.sources,
                     self.edges.flip(1),self.flags)
        self.assertGreater(float((a-b).abs().max()),1e-7)

    def test_self_mask_and_invalid_rank(self):
        mask=torch.tensor([[True,False]]*7)
        _,mix,_=self.model(self.visual,self.context,self.target,self.sources,
                           self.edges,self.flags,mask,return_aux=True)
        self.assertTrue(torch.equal(mix[:,1],torch.zeros(7)))
        with self.assertRaises(ValueError):
            ConceptLoRAComposer(12,16,hidden=32,rank=7)
        with self.assertRaises(ValueError):
            self.model(self.visual,self.context,self.target,self.sources,
                       self.edges,self.flags,torch.zeros_like(mask))


if __name__=="__main__":
    unittest.main()
