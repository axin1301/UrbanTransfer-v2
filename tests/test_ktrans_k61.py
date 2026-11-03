import unittest
import torch
from urban_transfer.ktrans_normalized import NormalizedExpertBank
from urban_transfer.ktrans_k5b import RegionTaskComposer
from urban_transfer.ktrans_k61 import AnchoredRegionalGate,build_source_transfer_routes


class AnchoredGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
        self.sources=['digital_modernization_tmi','education_accessibility',
                      'food_accessibility','nighttime_light_intensity','population_count']
        self.target='digital_total_cell_intensity'
        names=self.sources+[self.target]
        self.vectors={t:torch.randn(16) for t in names}
        stats={t:(float(i+1),.5+float(i)/4) for i,t in enumerate(self.sources)}
        self.bank=NormalizedExpertBank(8,self.vectors,self.sources,stats,hidden_dim=16)
        self.composer=RegionTaskComposer(self.bank,self.vectors,rank=4,adapter_cap=.5,
                                         adapter_strength=0.)
        routes=build_source_transfer_routes(self.vectors,self.sources,stats,
                                             [0.,1.,0.,0.],.15)
        self.model=AnchoredRegionalGate(self.composer,[.1,.2,.3,.2,.2],routes)
        self.batch={'image':torch.randn(5,8),'context':torch.randn(5,4),
                    'task_name':[self.target]*5}

    def test_initial_gate_and_convex_route(self):
        prediction,aux=self.model(self.batch,return_aux=True)
        self.assertEqual(prediction.shape,(5,))
        self.assertTrue(torch.allclose(aux['gate'],torch.full((5,),.1192029),atol=1e-6))
        self.assertTrue(torch.allclose(aux['weights'].sum(-1),torch.ones(5),atol=1e-6))
        self.assertTrue((aux['weights']>=0).all())

    def test_source_pseudotarget_own_expert_masked(self):
        batch=dict(self.batch,task_name=self.sources)
        _,aux=self.model(batch,exclude_own=True,return_aux=True)
        self.assertTrue(torch.allclose(torch.diag(aux['anchor']),torch.zeros(5),atol=1e-6))
        self.assertLess(float(torch.diag(aux['weights']).abs().max()),1e-5)
        self.assertTrue(torch.allclose(aux['weights'].sum(-1),torch.ones(5),atol=1e-6))

    def test_gradients_only_update_gate(self):
        batch=dict(self.batch,task_name=self.sources)
        _,aux=self.model(batch,exclude_own=True,return_aux=True)
        loss=aux['spatial'].square().mean()
        loss.backward()
        self.assertIsNotNone(self.model.gate.weight.grad)
        self.assertTrue(all(p.grad is None for p in self.composer.parameters()))

    def test_unseen_source_restrictions(self):
        with self.assertRaises(ValueError):
            self.model(self.batch,exclude_own=True)
        with self.assertRaises(ValueError):
            self.model(dict(self.batch,task_name=self.sources),exclude_own=False)

    def test_zero_gate_recovers_transfer(self):
        with torch.no_grad():
            self.model.gate.bias.fill_(-35.)
            _,aux=self.model(self.batch,return_aux=True)
        self.assertTrue(torch.allclose(aux['weights'],aux['anchor'],atol=1e-6))

if __name__=='__main__':unittest.main()
