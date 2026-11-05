"""CPU regression tests for source-only grouped MLLM hidden diagnostics."""
import unittest
import numpy as np
import torch

from scripts.audit_k7_m61_grouped_hidden import (
    spans,statistics,token_pair_metrics,analyze)
from urban_transfer.k7_m61_router import StructuredMLLMRouter
from urban_transfer.k7_mllm_hidden import TASKS


class M61GroupedAuditTests(unittest.TestCase):
    def test_token_group_boundaries(self):
        result=spans((3,4,5))
        self.assertEqual(result["visual"],slice(0,3))
        self.assertEqual(result["definition"],slice(3,7))
        self.assertEqual(result["relation"],slice(7,12))
        with self.assertRaises(ValueError):
            spans((3,0,5))

    def test_identical_and_distinct_tokens(self):
        torch.manual_seed(42)
        a=torch.randn(6,4,16)
        same=token_pair_metrics(a,a)
        self.assertAlmostEqual(same["pooled_cosine"]["mean"],1,places=5)
        self.assertAlmostEqual(same["relative_frobenius_difference"]["mean"],0,places=6)
        distinct=token_pair_metrics(a,a.roll(1,0))
        self.assertGreater(distinct["relative_frobenius_difference"]["mean"],.01)

    def test_same_image_task_pairs_are_group_specific(self):
        torch.manual_seed(42)
        model=StructuredMLLMRouter(12,16,hidden=32,
                                    group_counts=(3,4,5)).eval()
        rows=np.arange(5)
        cache={(i,task):torch.randn(12,16)
               for i in rows for task in TASKS}
        snapshot={k:v.detach().clone() for k,v in model.state_dict().items()}
        report=analyze(cache,rows,model,(3,4,5),batch_size=2)
        self.assertEqual(len(report),3)
        for parts in report.values():
            for group in ("visual","definition","relation"):
                self.assertIn("raw",parts[group])
                self.assertIn("student_projected",parts[group])
                self.assertTrue(np.isfinite(
                    parts[group]["raw"]["pooled_cosine"]["mean"]))
            self.assertIn("student_cross_attended",parts)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k])
                            for k,v in snapshot.items()))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            statistics([np.nan])


if __name__=="__main__":
    unittest.main()
