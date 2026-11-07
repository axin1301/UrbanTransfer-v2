"""CPU-only tests for frozen M6.4 source-only post-hoc calibration."""
import unittest

import numpy as np
import torch

from scripts.run_k7_m64_source_calibration import (
    apply_calibration,fit_source_affine,score_pair,semantic_weights,
    SHRINK,SLOPE_MIN,SLOPE_MAX)
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK


class SourceCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.names=[TRAIN_TASKS[0]]*6+[TRAIN_TASKS[1]]*6+[HELDOUT_TASK]*4
        self.pred=np.array([
            -1.5,-1.,-.5,.5,1.,1.5,
            -2.,-1.,-.25,.25,1.,2.,
            -2.,-1.,1.,2.],dtype=float)
        self.truth=np.array([
            -1.2,-.8,-.3,.4,.8,1.2,
            -1.5,-.7,-.1,.2,.9,1.5,
            10.,10.,10.,10.],dtype=float)
        self.stats={t:(0.,1.) for t in TRAIN_TASKS}
        self.train=np.arange(12)

    def test_target_labels_never_fit_calibration(self):
        c1=fit_source_affine(
            self.pred,self.truth,self.names,self.train,self.stats)
        wrong=self.truth.copy()
        wrong[12:]=1e12
        c2=fit_source_affine(
            self.pred,wrong,self.names,self.train,self.stats)
        self.assertEqual(c1,c2)
        for task in TRAIN_TASKS:
            self.assertGreaterEqual(c1[task]["slope"],SLOPE_MIN)
            self.assertLessEqual(c1[task]["slope"],SLOPE_MAX)

    def test_semantic_weights_use_only_frozen_vectors(self):
        vectors={
            TRAIN_TASKS[0]:torch.tensor([1.,0.]),
            TRAIN_TASKS[1]:torch.tensor([0.,1.]),
            HELDOUT_TASK:torch.tensor([.8,.2]),
        }
        w=semantic_weights(vectors)
        self.assertAlmostEqual(sum(w.values()),1.,places=6)
        self.assertGreater(w[TRAIN_TASKS[0]],w[TRAIN_TASKS[1]])

    def test_target_affine_is_source_weighted_and_pearson_preserving(self):
        coefficients=fit_source_affine(
            self.pred,self.truth,self.names,self.train,self.stats)
        weights={TRAIN_TASKS[0]:.6,TRAIN_TASKS[1]:.4}
        ids=np.arange(12,16)
        corrected=apply_calibration(
            self.pred,ids,self.names,coefficients,weights)
        a=sum(weights[t]*coefficients[t]["slope"] for t in TRAIN_TASKS)
        b=sum(weights[t]*coefficients[t]["bias"] for t in TRAIN_TASKS)
        self.assertTrue(np.allclose(corrected,a*self.pred[ids]+b))
        self.assertAlmostEqual(
            np.corrcoef(corrected,self.pred[ids])[0,1],1.,places=12)

    def test_source_val_not_needed_to_fit_or_choose(self):
        a=fit_source_affine(
            self.pred,self.truth,self.names,self.train,self.stats)
        p=self.pred.copy()
        y=self.truth.copy()
        p[12:]=10000.
        y[12:]=-10000.
        b=fit_source_affine(p,y,self.names,self.train,self.stats)
        self.assertEqual(a,b)

    def test_score_pair_reports_mean_bias_and_centered_error(self):
        coefficients=fit_source_affine(
            self.pred,self.truth,self.names,self.train,self.stats)
        weights={TRAIN_TASKS[0]:.6,TRAIN_TASKS[1]:.4}
        for mode in ("original","calibrated"):
            result=score_pair(
                "SCUI",mode,np.arange(12,16),self.pred,self.names,
                self.truth,self.stats,(0.,1.),coefficients,weights)
            self.assertEqual(result["n_rows"],4)
            self.assertAlmostEqual(
                result["mse_log1p"],
                result["mean_bias_log1p"]**2+
                result["centered_error_mse"],places=12)

    def test_inconsistent_indices_cannot_learn_target(self):
        with self.assertRaises(ValueError):
            fit_source_affine(
                self.pred,self.truth,self.names,np.arange(12,16),
                self.stats)


if __name__=="__main__":
    unittest.main()
