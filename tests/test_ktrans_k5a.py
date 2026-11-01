import unittest
import numpy as np
import torch
from scripts.run_ktrans_k5a import features,fit_ridge,softmax,safe_corr

class K5ATest(unittest.TestCase):
    def test_router_features(self):
        a=torch.tensor([1.,0.]);b=torch.tensor([0.,1.])
        f=features(a,b,(2.,3.))
        self.assertEqual(f.shape,(4,))
        self.assertAlmostEqual(f[1],0.,places=6)
    def test_ridge_is_finite(self):
        X=np.array([[1.,.1,1.,.2],[1.,.9,2.,.5],
                    [1.,.3,.8,.4],[1.,.8,1.5,.2]])
        y=np.array([.1,.9,.2,.8])
        beta=fit_ridge(X,y,1.)
        self.assertTrue(np.isfinite(beta).all())
        self.assertAlmostEqual(float(softmax(X@beta,.15).sum()),1.)
    def test_spatial_supervision_affine_invariant(self):
        y=np.arange(1.,10.)
        z=3*y+99
        self.assertAlmostEqual(safe_corr(y,z),1.)
        self.assertEqual(safe_corr(np.ones(4),np.arange(4)),0.)

if __name__=='__main__':unittest.main()
