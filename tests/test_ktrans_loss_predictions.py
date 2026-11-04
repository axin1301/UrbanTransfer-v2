import unittest
import numpy as np
from scripts.diagnose_ktrans_loss_predictions import summary_metrics

class PredictionShiftTests(unittest.TestCase):
    def test_exact_mixture_identity(self):
        global_pred=np.array([1.,3.,5.])
        regional=np.array([2.,1.,7.])
        g0=np.array([.1,.2,.3])
        g1=np.array([.2,.3,.4])
        p0=global_pred+g0*(regional-global_pred)
        p1=global_pred+g1*(regional-global_pred)
        stats=summary_metrics(g1,global_pred,regional,g0,p0,p1)
        self.assertLess(stats['identity_error'],1e-12)
        self.assertAlmostEqual(stats['gate_delta_mean_abs'],.1)
        self.assertGreater(stats['pred_delta_mean_abs'],0.)

    def test_no_route_headroom_means_no_prediction_change(self):
        x=np.array([1.,2.,3.])
        stats=summary_metrics(np.array([.8,.9,.7]),x,x,
                  np.array([.1,.1,.1]),x,x)
        self.assertEqual(stats['route_gap_std'],0.)
        self.assertEqual(stats['pred_delta_mean_abs'],0.)
        self.assertEqual(stats['identity_error'],0.)

    def test_reject_unpaired(self):
        with self.assertRaises(ValueError):
            summary_metrics([.1],[1,2],[1,2],[.2],[1,2],[1,2])

if __name__=='__main__':unittest.main()
