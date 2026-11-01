import unittest
import numpy as np
from scripts.diagnose_k5b0_complementarity import diagnose

class ComplementarityTests(unittest.TestCase):
    def test_redundant_and_complementary_experts(self):
        y=np.array([0.,1.,3.,2.,5.,4.,7.,6.])
        z=np.stack([y, y, -y, y*.6+np.roll(y,1)*.4])
        rows,pairs=diagnose(z,y,['good1','good2','bad','mixed'],'dummy','meta_SCUI')
        self.assertEqual(len(rows),4)
        self.assertEqual(len(pairs),6)
        bad=next(r for r in rows if r['expert']=='bad')
        self.assertLess(bad['delta_pearson'],0)
        self.assertAlmostEqual(next(p['prediction_corr'] for p in pairs
            if p['expert_a']=='good1' and p['expert_b']=='good2'),1.,places=6)

if __name__=='__main__':unittest.main()
