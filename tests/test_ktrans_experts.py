import unittest
import torch
from urban_transfer.ktrans_experts import SemanticExpertBank

class ExpertBankTests(unittest.TestCase):
    def setUp(self):
        self.vectors={'source_a':torch.tensor([1.,0.,0.]),
                      'source_b':torch.tensor([0.,1.,0.]),
                      'heldout':torch.tensor([.9,.1,0.])}
        self.model=SemanticExpertBank(8,self.vectors,['source_a','source_b'],
                                       hidden_dim=16,temperature=.15)
    def test_semantic_weights_and_no_heldout_expert(self):
        self.assertEqual(len(self.model.experts),2)
        self.assertGreater(self.model.routing[self.model.task_to_all['heldout'],0],
                           self.model.routing[self.model.task_to_all['heldout'],1])
        self.assertTrue(torch.allclose(self.model.routing.sum(-1),
                                       torch.ones(len(self.vectors)),atol=1e-6))
    def test_supervision_only_trains_own_expert(self):
        batch={'image':torch.randn(4,8),'context':torch.randn(4,4),
               'task_name':['source_a']*4}
        pred=self.model(batch,None)
        self.assertEqual(pred.shape,(4,))
        pred.square().mean().backward()
        self.assertGreater(sum(x.grad.abs().sum().item() for x in self.model.experts[0].parameters()),0)
        self.assertTrue(all(x.grad is None or x.grad.abs().sum()==0
                            for x in self.model.experts[1].parameters()))
    def test_unseen_prediction_is_weighted_sum(self):
        batch={'image':torch.randn(2,8),'context':torch.randn(2,4),
               'task_name':['heldout','heldout']}
        p=self.model(batch,None)
        with torch.no_grad():
            x=torch.cat([batch['image'],batch['context']],dim=-1)
            h=self.model.shared(x)
            expected=sum(self.model.routing[self.model.task_to_all['heldout'],j]
                         *expert(h).squeeze(-1) for j,expert in enumerate(self.model.experts))
        self.assertTrue(torch.allclose(p,expected,atol=1e-6))

if __name__=='__main__':unittest.main()
