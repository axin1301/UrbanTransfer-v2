import unittest
import pandas as pd
import torch
from urban_transfer.ktrans_normalized import NormalizedExpertBank,source_moments

class K3ANormalizationTests(unittest.TestCase):
    def setUp(self):
        self.df=pd.DataFrame([
            {'indicator_column':'a','indicator_value_log1p':v}
            for v in (2.,4.,6.)
        ]+[
            {'indicator_column':'b','indicator_value_log1p':v}
            for v in (10.,12.,14.)
        ]+[
            {'indicator_column':'held','indicator_value_log1p':999.}
        ])
        self.vec={'a':torch.tensor([1.,0.]),'b':torch.tensor([0.,1.]),
                  'held':torch.tensor([.5,.5])}
        self.stats=source_moments(self.df[self.df.indicator_column!='held'],['a','b'])
    def test_source_only_moments(self):
        self.assertEqual(set(self.stats),{'a','b'})
        self.assertAlmostEqual(self.stats['a'][0],4)
        self.assertAlmostEqual(self.stats['b'][0],12)
        self.assertGreater(self.stats['a'][1],0)
    def test_no_heldout_label_in_model(self):
        model=NormalizedExpertBank(4,self.vec,['a','b'],self.stats,hidden_dim=8)
        batch={'image':torch.zeros(2,4),'context':torch.zeros(2,4),
               'task_name':['a','held']}
        with torch.no_grad():
            for expert in model.experts:
                for p in expert.parameters():p.zero_()
            model.set_routing_mode('uniform')
            y=model(batch)
        self.assertAlmostEqual(float(y[0]),4.,places=5)
        self.assertAlmostEqual(float(y[1]),8.,places=5)
        self.assertTrue(torch.isfinite(y).all())
    def test_seen_task_supervises_own_expert(self):
        model=NormalizedExpertBank(4,self.vec,['a','b'],self.stats,hidden_dim=8)
        batch={'image':torch.randn(3,4),'context':torch.randn(3,4),
               'task_name':['a']*3}
        model(batch).square().mean().backward()
        self.assertTrue(all(p.grad is None or p.grad.abs().sum()==0 for p in model.experts[1].parameters()))
    def test_reject_missing_moments(self):
        with self.assertRaises(ValueError):
            NormalizedExpertBank(4,self.vec,['a','b'],{'a':self.stats['a']},hidden_dim=8)

if __name__=='__main__':unittest.main()
