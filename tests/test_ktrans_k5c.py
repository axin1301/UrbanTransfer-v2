import unittest
import torch
from urban_transfer.ktrans_k5c import IndependentScaleRouter,scale_features

class K5CScaleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.vectors={name:torch.randn(32) for name in ('a','b','c','d','e','held')}
        self.stats={n:(float(i+1),float(i+2)) for i,n in enumerate(('a','b','c','d','e'))}

    def test_source_only_predictions(self):
        router=IndependentScaleRouter(self.vectors,self.stats)
        mu,std=router.predict('held')
        self.assertTrue(-100<mu<100)
        self.assertGreater(std,0)
        self.assertEqual(router.coef.shape,(3,2))
        with self.assertRaises(ValueError):router.predict('a')

    def test_leave_one_out_does_not_use_target_moments(self):
        a=scale_features('a',['b','c','d','e'],self.vectors,self.stats)
        changed=dict(self.stats);changed['a']=(10000.,10000.)
        b=scale_features('a',['b','c','d','e'],self.vectors,changed)
        self.assertTrue((a==b).all())

    def test_scale_change_preserves_spatial_pearson(self):
        x=torch.arange(1.,8.).numpy()
        y=2*x+.25
        import numpy as np
        self.assertAlmostEqual(float(np.corrcoef(y,1.5+3*x)[0,1]),
                               float(np.corrcoef(y,8.+.7*x)[0,1]),places=10)

if __name__=='__main__':unittest.main()
