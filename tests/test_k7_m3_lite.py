"""M3 Lite: parameter-level LoRA composition and two-edge identity regression tests."""
import unittest
import torch
from urban_transfer.k7_lora_composer import RelationLoRAComposer

class M3LiteTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=RelationLoRAComposer(12,16,hidden=32,rank=4)
        self.x=torch.randn(6,12)
        self.context=torch.randn(6,4)
        self.target=torch.randn(6,16)
        self.sources=torch.randn(2,16)
        self.relations=torch.randn(6,2,16)
        self.flags=torch.zeros(6,2)

    def invoke(self,relations=None,mask=None,aux=False):
        return self.model(self.x,self.context,self.target,self.sources,
            self.relations if relations is None else relations,
            self.flags,mask,return_aux=aux)

    def test_shape_gradient_both_source_loras_and_rank_gate(self):
        # LoRA B starts at zero, and receives its first gradient step before
        # the gate becomes trainable (standard LoRA initialization).
        opt=torch.optim.AdamW(self.model.parameters(),lr=.005)
        for _ in range(2):
            opt.zero_grad()
            y,mix,gates=self.invoke(aux=True)
            self.assertEqual(tuple(y.shape),(6,))
            self.assertEqual(tuple(mix.shape),(6,2))
            self.assertEqual(tuple(gates.shape),(6,2,4))
            self.assertTrue(torch.allclose(mix.sum(-1),torch.ones(6),atol=1e-6))
            y.square().mean().backward()
            self.assertGreater(float(self.model.source_b.grad.abs().sum()),0.)
            opt.step()
        opt.zero_grad()
        self.invoke().square().mean().backward()
        self.assertGreater(float(self.model.rank_gate.weight.grad.abs().sum()),0.)

    def test_swapping_relation_tokens_changes_predictions(self):
        # Ensure adapter B is nonzero, since initial LoRA B=0 by design.
        with torch.no_grad():
            self.model.source_b.normal_(0.,.1)
        left=self.invoke()
        right=self.invoke(self.relations.flip(1))
        self.assertGreater(float((left-right).abs().max()),1e-7)

    def test_masked_self_path_is_exactly_excluded(self):
        mask=torch.tensor([[True,False]]*6)
        _,weights,_=self.invoke(mask=mask,aux=True)
        self.assertTrue(torch.equal(weights[:,1],torch.zeros(6)))
        with self.assertRaises(ValueError):
            self.invoke(mask=torch.zeros_like(mask))

    def test_image_conditioned_source_weight(self):
        with torch.no_grad():
            self.model.source_b.normal_(0.,.1)
        _,w1,_=self.invoke(aux=True)
        x=self.x.clone()
        x[0]*=8
        _,w2,_=self.model(x,self.context,self.target,self.sources,
                          self.relations,self.flags,return_aux=True)
        self.assertGreater(float((w1[0]-w2[0]).abs().max()),1e-8)

if __name__=="__main__":
    unittest.main()
