import unittest
import numpy as np
from scripts.analyze_ktrans_predictions import summaries

class KTransDiagnosticTests(unittest.TestCase):
    def test_perfect_prediction(self):
        r=summaries([1,2,3],[1,2,3])
        self.assertAlmostEqual(r['mse'],0)
        self.assertAlmostEqual(r['r2'],1)
        self.assertAlmostEqual(r['pearson'],1)
        self.assertAlmostEqual(r['bias'],0)
        self.assertAlmostEqual(r['oracle_mean_mse'],2/3)
    def test_constant_predictions(self):
        r=summaries([1,2,3],[2,2,2])
        self.assertIsNone(r['pearson'])
        self.assertAlmostEqual(r['pred_std'],0)
        self.assertAlmostEqual(r['mse'],r['oracle_mean_mse'])
        self.assertAlmostEqual(r['spatial_gain_vs_pred_mean_constant'],0)
    def test_mean_bias(self):
        r=summaries([0,2],[1,3])
        self.assertAlmostEqual(r['bias'],1)
        self.assertAlmostEqual(r['mse'],1)
        self.assertAlmostEqual(r['r2'],0)

if __name__=='__main__':unittest.main()
