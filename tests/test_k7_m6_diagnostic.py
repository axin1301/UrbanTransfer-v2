"""CPU tests for K7 M6 read-only hidden/router diagnostic."""
import unittest
import numpy as np
import torch

from scripts.diagnose_k7_m6_hidden import summarize_hidden,student_diagnostic,summary
from urban_transfer.k7_mllm_router import MLLMHiddenRouter


class DiagnosticTests(unittest.TestCase):
    def test_teacher_similarity_and_uniform_case(self):
        torch.manual_seed(42)
        target=torch.randn(6,8,32)
        sources=torch.randn(6,2,8,32)
        cos,teacher,stats=summarize_hidden(sources,target,.2)
        self.assertEqual(cos.shape,(6,2))
        self.assertEqual(teacher.shape,(6,2))
        self.assertTrue(np.allclose(teacher.sum(-1),1))
        self.assertIn("teacher_pop",stats)
        same=torch.stack((target,target),dim=1)
        _,uniform,_=summarize_hidden(same,target,.2)
        self.assertTrue(np.allclose(uniform,.5))

    def test_frozen_model_and_mismatch(self):
        torch.manual_seed(42)
        model=MLLMHiddenRouter(12,32,hidden=32).eval()
        image=torch.randn(7,12)
        context=torch.randn(7,4)
        target=torch.randn(7,8,32)
        before=[p.clone() for p in model.parameters()]
        w,p,sw,sp=student_diagnostic(model,image,context,target,None,3)
        self.assertEqual(w.shape,(7,2))
        self.assertEqual(p.shape,(7,))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,model.parameters())))
        self.assertGreater(float(np.abs(p-sp).max()),0)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            summary([float("nan")])


if __name__=="__main__":
    unittest.main()
