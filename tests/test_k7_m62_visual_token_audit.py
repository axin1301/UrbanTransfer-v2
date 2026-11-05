"""CPU-only regression tests for M6.2 frozen visual token intervention."""
import unittest

import numpy as np
import torch

from scripts.audit_k7_m62_visual_tokens import (
    evaluate_visual_interventions,replace_visual_only,summary)
from urban_transfer.k7_m62_task_query import TaskQueriedRouter
from urban_transfer.k7_m1_knowledge import HELDOUT_TASK


class FrozenVisualM62Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=TaskQueriedRouter(
            12,24,hidden=32,heads=4,group_counts=(6,4,5)).eval()
        self.x=torch.randn(7,12)
        self.ctx=torch.randn(7,4)
        self.y=torch.randn(7)
        self.cache={(i,HELDOUT_TASK):torch.randn(15,24) for i in range(7)}

    def test_visual_only_swap_preserves_text_unchanged(self):
        original=torch.stack([self.cache[(i,HELDOUT_TASK)] for i in (0,1)])
        donor=torch.stack([self.cache[(i,HELDOUT_TASK)] for i in (1,2)])
        original_before=original.clone()
        swapped=replace_visual_only(original,donor,6,"swap_visual")
        self.assertTrue(torch.equal(swapped[:,6:,:],original[:,6:,:]))
        self.assertTrue(torch.equal(swapped[:,:6,:],donor[:,:6,:]))
        zeroed=replace_visual_only(original,donor,6,"zero_visual")
        self.assertTrue(torch.equal(zeroed[:,:6,:],torch.zeros_like(zeroed[:,:6,:])))
        self.assertTrue(torch.equal(zeroed[:,6:,:],original[:,6:,:]))
        self.assertTrue(torch.equal(original,original_before))
        self.assertTrue(torch.equal(
            replace_visual_only(original,donor,6,"matched"),original))

    def test_only_visual_change_task_query_is_invariant(self):
        orig=torch.stack([self.cache[(i,HELDOUT_TASK)] for i in range(3)])
        wrong=orig.roll(1,0)
        with torch.inference_mode():
            a=self.model.condition(orig)[1]
            b=self.model.condition(
                replace_visual_only(orig,wrong,6,"swap_visual"))[1]
            c=self.model.condition(
                replace_visual_only(orig,None,6,"zero_visual"))[1]
        self.assertTrue(torch.allclose(a,b,atol=1e-6))
        self.assertTrue(torch.allclose(a,c,atol=1e-6))

    def test_inference_audit_no_weight_change_and_finite_results(self):
        checkpoint={k:v.detach().clone()
                    for k,v in self.model.state_dict().items()}
        with torch.inference_mode():
            report=evaluate_visual_interventions(
                self.model,self.x,self.ctx,self.y,
                np.arange(7,dtype=np.int64),self.cache,
                6,0.,1.,batch_size=1,device="cpu")
        self.assertEqual(report["matched"]["n_rows"],7)
        self.assertAlmostEqual(
            report["matched"]["abs_log1p_prediction_delta_vs_matched"]["mean"],
            0.,places=7)
        for key in ("matched","swap_visual","zero_visual"):
            self.assertTrue(np.isfinite(report[key]["mse_log1p"]))
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k])
                            for k,v in checkpoint.items()))

    def test_bad_inputs_fail(self):
        original=torch.randn(2,15,24)
        with self.assertRaises(ValueError):
            replace_visual_only(original,None,6,"swap_visual")
        with self.assertRaises(ValueError):
            replace_visual_only(original,None,6,"invalid")
        with self.assertRaises(ValueError):
            evaluate_visual_interventions(
                self.model,self.x,self.ctx,self.y,np.array([0]),
                self.cache,6,0.,1.,2,"cpu")
        with self.assertRaises(ValueError):
            summary([np.nan])


if __name__=="__main__":
    unittest.main()
