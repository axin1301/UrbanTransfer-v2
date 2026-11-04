import unittest
import numpy as np
import torch
from urban_transfer.ktrans_k7d import AnchoredKnowledgeRouter
from urban_transfer.ktrans_k7e import route_response
from scripts.run_ktrans_k7f_gate_scan import fixed_gate_prediction,evaluate_grid,GATES

class K7FGateCounterfactualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.net=AnchoredKnowledgeRouter(3).eval()
        self.k=torch.rand(8,36)
        self.p=torch.tensor([[.15,.35,.50]]*8)
        self.mask=torch.zeros_like(self.p,dtype=torch.bool)
        self.z=torch.randn(8,3)
        self.mean=torch.full((8,),.3)
        self.std=torch.full((8,),1.5)
        with torch.no_grad():
            self.response=route_response(self.net,self.k,self.p,self.mask,
                                         self.z,self.mean,self.std)
    def test_zero_gate_equals_frozen_baseline(self):
        pred,w=fixed_gate_prediction(self.response,self.z,self.mean,self.std,0.)
        self.assertTrue(torch.allclose(pred,self.response['baseline'],atol=1e-7))
        self.assertTrue(torch.allclose(w,self.response['prior'],atol=1e-7))
    def test_one_gate_equals_candidate(self):
        pred,w=fixed_gate_prediction(self.response,self.z,self.mean,self.std,1.)
        self.assertTrue(torch.allclose(w,self.response['candidate'],atol=1e-7))
        expected=self.mean+self.std*(self.response['candidate']*self.z).sum(-1)
        self.assertTrue(torch.allclose(pred,expected,atol=1e-7))
    def test_learned_gate_reproduces_checkpoint_forward(self):
        pred=self.response['prediction']
        manual=self.mean+self.std*(self.response['final']*self.z).sum(-1)
        self.assertTrue(torch.allclose(pred,manual,atol=1e-7))
    def test_grid_is_fixed_and_baseline_invariant(self):
        labels=np.arange(8,dtype=float)/8.
        rows=evaluate_grid(self.response,self.z,self.mean,self.std,labels,
                           'image_task','meta_UCUI')
        self.assertEqual(len(rows),len(GATES)+1)
        self.assertEqual(rows[0]['gate'],0.)
        self.assertAlmostEqual(rows[0]['delta_mse_vs_fixed_zero'],0.)
        self.assertEqual(rows[-1]['gate_mode'],'trained')
        self.assertTrue(all('pearson' in r for r in rows))
    def test_invalid_gate_rejected(self):
        with self.assertRaises(ValueError):
            fixed_gate_prediction(self.response,self.z,self.mean,self.std,1.2)

if __name__=='__main__':unittest.main()
