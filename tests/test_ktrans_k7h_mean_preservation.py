import unittest
import numpy as np
from urban_transfer.ktrans_k7h import preserve_mean,mean_preserving_error_identity
from urban_transfer.ktrans_k7g import error_decomposition

class K7HMeanPreservationTests(unittest.TestCase):
    def test_no_labels_required_for_projection(self):
        baseline=np.array([1.,3.,2.,4.])
        prediction=np.array([2.,5.,4.,7.])
        tasks=['task_a','task_a','task_b','task_b']
        adjusted,audit=preserve_mean(baseline,prediction,tasks)
        np.testing.assert_allclose(adjusted,[.5,3.5,.5,5.5])
        self.assertEqual(len(audit),2)
        self.assertAlmostEqual(audit[0]['label_free_shift_subtracted'],1.5)
        self.assertAlmostEqual(audit[1]['label_free_shift_subtracted'],2.5)

    def test_centered_mse_unchanged_and_bias_preserved(self):
        baseline=np.array([1.,2.,3.,4.,3.,4.])
        raw=np.array([2.,4.,6.,8.,5.,6.])
        y=np.array([.5,1.5,3.5,3.,4.,4.5])
        tasks=['a','a','a','a','b','b']
        adjusted,_=preserve_mean(baseline,raw,tasks)
        self.assertTrue(mean_preserving_error_identity(y,baseline,raw,adjusted,tasks))
        for task in ('a','b'):
            ix=np.array(tasks)==task
            orig=error_decomposition(y[ix],raw[ix],baseline[ix])
            fixed=error_decomposition(y[ix],adjusted[ix],baseline[ix])
            self.assertAlmostEqual(fixed['delta_bias_squared'],0.)
            self.assertAlmostEqual(fixed['delta_centered_mse'],orig['delta_centered_mse'])
            self.assertAlmostEqual(fixed['delta_mse_vs_k5a'],orig['delta_centered_mse'])

    def test_singleton_group_does_not_leak_other_task(self):
        base=np.array([1.,2.,4.])
        raw=np.array([10.,4.,5.])
        corrected,_=preserve_mean(base,raw,['a','b','b'])
        self.assertAlmostEqual(corrected[0],base[0])
        self.assertAlmostEqual(corrected[1:].mean(),base[1:].mean())

    def test_fixed_zero_is_identity(self):
        b=np.array([0.,1.,3.])
        fixed,audit=preserve_mean(b,b,['x','x','x'])
        np.testing.assert_array_equal(fixed,b)
        self.assertEqual(audit[0]['label_free_shift_subtracted'],0.)

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            preserve_mean([1.,2.],[1.],['x','x'])
        with self.assertRaises(ValueError):
            preserve_mean([1.],[float('nan')],['x'])
        with self.assertRaises(ValueError):
            preserve_mean([1.],[2.],[''])
if __name__=='__main__':unittest.main()
