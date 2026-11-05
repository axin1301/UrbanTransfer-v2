"""CPU regression tests for frozen M6.2 calibration/stage audit."""
import unittest

import numpy as np
import torch

from scripts.audit_k7_m62_calibration_attention import (
    mse_decomposition,dev_calibration,representation_pair,
    same_image_stage_differences)
from urban_transfer.k7_m62_task_query import TaskQueriedRouter
from urban_transfer.k7_mllm_hidden import TASKS


class M62CalibrationAuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=TaskQueriedRouter(
            12,24,hidden=32,heads=4,group_counts=(6,4,5)).eval()
        self.x=torch.randn(7,12)
        self.geo=torch.randn(7,4)
        self.y=torch.randn(7)
        self.cache={(i,t):torch.randn(15,24)
                    for i in range(7) for t in TASKS}

    def test_exact_mse_decomposition_with_nontrivial_covariance(self):
        pred=np.array([1.4,3.3,-1.,2.,8.5,3.],dtype=float)
        truth=np.array([2.,0.,-2.5,5.,4.,1.],dtype=float)
        d=mse_decomposition(pred,truth)
        self.assertAlmostEqual(d["mse"],sum([
            d["mean_bias_squared"],d["std_mismatch_squared"],
            d["correlation_component"]]),places=12)
        self.assertAlmostEqual(
            d["pearson"],float(np.corrcoef(pred,truth)[0,1]),places=12)

    def test_constant_predictions_have_valid_components(self):
        truth=[1.,2.,3.,4.]
        result=mse_decomposition([2.,2.,2.,2.],truth)
        self.assertIsNone(result["pearson"])
        self.assertAlmostEqual(result["mse"],1.5)
        self.assertAlmostEqual(result["mse_reconstructed"],1.5)

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValueError):
            mse_decomposition([0.,np.nan],[0.,1.])
        with self.assertRaises(ValueError):
            mse_decomposition([1.],[1.])

    def test_query_attention_fusion_are_distinct_outputs(self):
        indices=np.arange(7,dtype=np.int64)
        weights={k:v.detach().clone()
                 for k,v in self.model.state_dict().items()}
        result=same_image_stage_differences(
            self.model,indices,self.cache,batch_size=3,device="cpu")
        self.assertEqual(len(result),3)
        for pair in result.values():
            for stage in ("query","attention_output","fusion_output"):
                self.assertTrue(np.isfinite(pair[stage]["cosine"]["mean"]))
                self.assertTrue(np.isfinite(pair[stage]["relative_l2"]["mean"]))
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k])
                            for k,v in weights.items()))

    def test_frozen_dev_decomposition_parity(self):
        indices=np.arange(7,dtype=np.int64)
        report=dev_calibration(self.model,self.x,self.geo,self.y,indices,
                               self.cache,mean=1.,scale=2.,
                               batch_size=2,device="cpu")
        self.assertEqual(len(report),4)
        for mode,item in report.items():
            self.assertAlmostEqual(
                item["mse"],item["mean_bias_squared"]+
                item["std_mismatch_squared"]+item["correlation_component"],
                places=9)
        with torch.inference_mode():
            z=self.model(self.x,self.geo,torch.stack([
                self.cache[(i,TASKS[-1])] for i in range(7)]))
        actual=np.mean(((1.+2.*z.numpy())-self.y.numpy())**2)
        self.assertAlmostEqual(report["normal"]["mse"],actual,places=5)

    def test_same_image_representation_comparison(self):
        a=torch.randn(5,16)
        d=representation_pair(a,a.clone())
        self.assertAlmostEqual(d["cosine"]["mean"],1.,places=6)
        self.assertAlmostEqual(d["relative_l2"]["mean"],0.,places=6)
        with self.assertRaises(ValueError):
            representation_pair(a,a[:2])


if __name__=="__main__":
    unittest.main()
