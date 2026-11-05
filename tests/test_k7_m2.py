"""Fast CPU shape/gradient/edge-identity tests for directed K7 M2."""
import unittest
import torch

from urban_transfer.k7_directed_fusion import DirectedEdgeFusionRegressor
from scripts.run_k7_m2 import build_tokens
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS, HELDOUT_TASK


class K7M2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = DirectedEdgeFusionRegressor(12,16,hidden=32,rank=4)
        self.x = torch.randn(7,12)
        self.ctx = torch.randn(7,4)
        self.t = torch.randn(7,16)
        self.s = torch.randn(2,16)
        self.r = torch.randn(7,2,16)
        self.flags = torch.zeros(7,2)

    def test_one_forward_and_gradients(self):
        y,attention = self.model(self.x,self.ctx,self.t,self.s,self.r,
                                 self.flags,return_attention=True)
        self.assertEqual(tuple(y.shape),(7,))
        self.assertEqual(tuple(attention.shape),(7,2))
        self.assertTrue(torch.allclose(attention.sum(-1),torch.ones(7)))
        y.square().mean().backward()
        self.assertGreater(float(self.model.hyper[-1].weight.grad.abs().sum()),0.)
        self.assertGreater(float(self.model.source_experts[0][1].weight.grad.abs().sum()),0.)

    def test_source_order_is_not_invariant(self):
        original = self.model(self.x,self.ctx,self.t,self.s,self.r,self.flags)
        # Text swapping without expert swapping is NOT a neutral operation.
        changed = self.model(self.x,self.ctx,self.t,self.s,self.r.flip(1),
                             self.flags)
        self.assertGreater(float((original-changed).abs().max()),1e-7)

    def test_self_mask_removes_path_without_nan(self):
        mask = torch.ones(7,2,dtype=torch.bool)
        mask[:,0] = False
        _,weights = self.model(self.x,self.ctx,self.t,self.s,self.r,self.flags,
                               mask,return_attention=True)
        self.assertTrue(torch.all(weights[:,0] == 0))
        with self.assertRaises(ValueError):
            self.model(self.x,self.ctx,self.t,self.s,self.r,self.flags,
                       torch.zeros_like(mask))

    def test_education_has_two_real_edges_train_has_self_edge(self):
        task_embeddings = {task:torch.randn(16)
                           for task in (*TRAIN_TASKS,HELDOUT_TASK)}
        cards = {(a,b):torch.randn(16) for a in TRAIN_TASKS
                 for b in (*TRAIN_TASKS,HELDOUT_TASK) if a!=b}
        rel,flags = build_tokens(HELDOUT_TASK,task_embeddings,cards)
        self.assertEqual(tuple(rel.shape),(2,16))
        self.assertTrue(torch.all(flags == 0))
        _,source_flags = build_tokens(TRAIN_TASKS[0],task_embeddings,cards)
        self.assertEqual(int(source_flags.sum()),1)


if __name__ == "__main__":
    unittest.main()
