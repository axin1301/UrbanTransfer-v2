import unittest
import torch
from urban_transfer.ktrans_k7d import AnchoredKnowledgeRouter
from urban_transfer.ktrans_k7e import route_response,summarise,expert_rows

class K7EResponseTests(unittest.TestCase):
    def test_reconstruction_and_masked_experts(self):
        torch.manual_seed(4)
        net=AnchoredKnowledgeRouter(4).eval()
        k=torch.rand(9,36)
        p=torch.tensor([[0.,.2,.3,.5]]*9)
        mask=torch.tensor([[True,False,False,False]]*9)
        experts=torch.randn(9,4)
        with torch.inference_mode():
            r=route_response(net,k,p,mask,experts,torch.zeros(9),torch.ones(9))
            expected,g=net(k,p,mask)
        self.assertTrue(torch.allclose(r['final'],expected,atol=1e-6))
        self.assertTrue(torch.allclose(r['gate'],g,atol=1e-6))
        self.assertTrue(torch.allclose(r['final'][:,0],torch.zeros(9)))
        s=summarise(k,r,experts,['t']*9)
        self.assertGreater(s['knowledge_varying_features'],0)
        self.assertGreaterEqual(s['candidate_prior_l1_mean'],0)
        self.assertEqual(len(expert_rows(r,['a','b','c','d'])),4)

    def test_baseline_has_no_changes(self):
        k=torch.zeros(5,36)
        prior=torch.tensor([[.2,.3,.5]]*5)
        expert=torch.randn(5,3)
        r=route_response(None,k,prior,torch.zeros_like(prior,dtype=torch.bool),
                         expert,torch.zeros(5),torch.ones(5))
        s=summarise(k,r,expert,['t']*5)
        self.assertEqual(s['prediction_delta_abs_mean'],0.)
        self.assertEqual(s['final_prior_l1_mean'],0.)
        self.assertEqual(s['gate_knowledge_corr_valid_dimensions'],0)

    def test_all_experts_masked_rejected(self):
        with self.assertRaises(ValueError):
            route_response(None,torch.zeros(2,36),torch.ones(2,3),
                           torch.ones(2,3,dtype=torch.bool),torch.ones(2,3),
                           torch.zeros(2),torch.ones(2))
if __name__=='__main__':unittest.main()
