import unittest
import numpy as np
import torch
from urban_transfer.ktrans_normalized import NormalizedExpertBank
from urban_transfer.ktrans_k5b import RegionTaskComposer
from urban_transfer.ktrans_k6 import UnifiedKTransMoE,VARIANTS,combine_prior,normalize_route

class K6Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        names=['population_count','nighttime_light_intensity',
               'education_accessibility','food_accessibility',
               'digital_modernization_tmi','digital_total_cell_intensity']
        vectors={s:torch.randn(24) for s in names}
        sources=sorted(names[:5])
        stats={s:(float(i+1),float(i+1)/2) for i,s in enumerate(sources)}
        bank=NormalizedExpertBank(8,vectors,sources,stats,hidden_dim=16)
        composer=RegionTaskComposer(bank,vectors,rank=4,adapter_cap=.5)
        self.model=UnifiedKTransMoE(composer,vectors,stats,
                                    'digital_total_cell_intensity',[1]*5)
        self.unseen={'image':torch.randn(4,8),'context':torch.randn(4,4),
                      'task_name':['digital_total_cell_intensity']*4}
        self.seen={**self.unseen,'task_name':['food_accessibility']*4}

    def test_k4_equals_frozen_bank(self):
        with torch.no_grad():
            p=self.model.forward_variant(self.unseen,'k4')
            base=self.model.bank(self.unseen)
        self.assertTrue(torch.allclose(p,base,atol=1e-5))

    def test_seen_task_uses_its_own_expert(self):
        with torch.no_grad():
            base=self.model.bank(self.seen)
            for name in VARIANTS:
                self.assertTrue(torch.allclose(
                    self.model.forward_variant(self.seen,name),base,atol=1e-5))

    def test_all_variants_finite(self):
        with torch.no_grad():
            for name in VARIANTS:
                result=self.model.forward_variant(self.unseen,name)
                self.assertEqual(result.shape,(4,))
                self.assertTrue(torch.isfinite(result).all())

    def test_transfer_prior_normalization(self):
        self.assertAlmostEqual(normalize_route([2,3,5]).sum(),1.)
        with self.assertRaises(ValueError):normalize_route([-1,2])
        w=combine_prior(torch.tensor([[.2,.8]]),torch.tensor([[.4,.6]]),
                        torch.tensor([.5,.5]))
        self.assertAlmostEqual(w.sum().item(),1.,places=6)
        self.assertGreater(w[0,1].item(),w[0,0].item())

    def test_scale_only_does_not_change_pearson(self):
        with torch.no_grad():
            a=self.model.forward_variant(self.unseen,'k5b2').numpy()
            b=self.model.forward_variant(self.unseen,'k5d').numpy()
        if a.std()>1e-7 and b.std()>1e-7:
            self.assertAlmostEqual(np.corrcoef(a,b)[0,1],1.,places=5)

if __name__=='__main__':unittest.main()
