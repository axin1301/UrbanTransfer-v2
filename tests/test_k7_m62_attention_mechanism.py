"""CPU tests for M6.2 frozen attention and bypass diagnostics."""
import unittest
import numpy as np
import torch

from urban_transfer.k7_m62_task_query import TaskQueriedRouter
from urban_transfer.k7_mllm_hidden import TASKS
from scripts.audit_k7_m62_attention_mechanism import (
    MODES,attention_diagnostics,masked_forward,normalized_entropy,
    js_divergence,partition_interventions,source_attention_pairs)


class M62AttentionMechanismTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=TaskQueriedRouter(
            12,24,hidden=32,heads=4,group_counts=(6,4,5)).eval()
        self.x=torch.randn(7,12)
        self.ctx=torch.randn(7,4)
        self.y=torch.randn(7)
        self.cache={(i,t):torch.randn(15,24)
                    for i in range(7) for t in TASKS}

    def test_normal_reconstructs_frozen_forward(self):
        tokens=torch.stack([self.cache[(i,TASKS[-1])]
                            for i in range(7)])
        with torch.no_grad():
            query,attended,fused,weights=attention_diagnostics(
                self.model,tokens)
            output,route=masked_forward(
                self.model,self.x,self.ctx,query,fused,attended,"normal")
            original,original_route,orig_q,orig_fused=self.model(
                self.x,self.ctx,tokens,return_aux=True)
        self.assertTrue(torch.allclose(output,original,atol=1e-5,rtol=1e-5))
        self.assertTrue(torch.allclose(route,original_route,atol=1e-5,rtol=1e-5))
        self.assertTrue(torch.allclose(query,orig_q,atol=1e-5))
        self.assertTrue(torch.allclose(fused,orig_fused,atol=1e-5))
        self.assertEqual(tuple(weights.shape),(7,4,6))
        self.assertTrue(torch.allclose(weights.sum(-1),
                        torch.ones(7,4),atol=1e-5))

    def test_pathway_interventions_are_independent(self):
        tokens=torch.stack([self.cache[(i,TASKS[-1])]
                            for i in range(7)])
        with torch.no_grad():
            query,attended,fused,_=attention_diagnostics(self.model,tokens)
            outputs={}
            for mode in MODES:
                outputs[mode]=masked_forward(
                    self.model,self.x,self.ctx,query,fused,attended,mode)
            self.assertEqual(len(outputs),4)
            for mode in MODES[1:]:
                self.assertTrue(torch.isfinite(outputs[mode][0]).all())
            with self.assertRaises(ValueError):
                masked_forward(self.model,self.x,self.ctx,
                               query,fused,attended,"invalid")

    def test_attention_entropy_js(self):
        same=torch.full((3,4,6),1/6)
        self.assertTrue(torch.allclose(normalized_entropy(same),
                         torch.ones(3),atol=1e-6))
        self.assertTrue(torch.allclose(js_divergence(same,same),
                         torch.zeros(3),atol=1e-7))
        different=torch.zeros((3,4,6))
        different[:,:,0]=1
        self.assertTrue(bool((js_divergence(same,different)>0).all()))

    def test_full_cpu_audit_is_read_only(self):
        before={k:v.detach().clone()
                for k,v in self.model.state_dict().items()}
        indices=np.arange(7,dtype=np.int64)
        dev=partition_interventions(self.model,self.x,self.ctx,self.y,
                                    indices,self.cache,0.,1.,3,"cpu")
        self.assertEqual(dev["modes"]["normal"]["n_rows"],7)
        self.assertAlmostEqual(dev["modes"]["normal"][
            "abs_prediction_change_log1p"]["mean"],0,places=7)
        same=source_attention_pairs(self.model,indices,
                                    self.cache,3,"cpu")
        self.assertEqual(len(same["same_image_cross_indicator_js"]),3)
        self.assertEqual(len(same["different_image_same_indicator_js"]),3)
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k])
                            for k,v in before.items()))


if __name__=="__main__":
    unittest.main()
