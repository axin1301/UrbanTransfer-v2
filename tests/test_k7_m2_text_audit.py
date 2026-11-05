"""No-training M2 text counterfactual audit logic tests."""
import unittest
import torch

from scripts.audit_k7_m2_text import (
    generic_relation_text, generic_cache_path, relation_conditions)
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS, HELDOUT_TASK


class M2TextAuditTests(unittest.TestCase):
    def test_swap_keeps_two_expert_slots_and_is_not_mean(self):
        a = torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        g = torch.tensor([[0.,0.,1.],[0.,0.,1.]])
        cases = relation_conditions(a,g)
        self.assertEqual(tuple(cases["correct"].shape),(2,3))
        self.assertTrue(torch.equal(cases["swapped"][0],a[1]))
        self.assertTrue(torch.equal(cases["swapped"][1],a[0]))
        self.assertFalse(torch.equal(cases["correct"],cases["swapped"]))
        self.assertTrue(torch.equal(cases["generic"],g))

    def test_generic_preserves_task_identity_without_mechanism(self):
        for source in TRAIN_TASKS:
            t=generic_relation_text(source, HELDOUT_TASK)
            self.assertIn(source,t)
            self.assertIn(HELDOUT_TASK,t)
            self.assertIn("unspecified",t)
            self.assertNotIn("Road connectivity",t)
        p=generic_cache_path("somewhere","Qwen/Qwen3-Embedding-0.6B")
        self.assertTrue(p.name.startswith("k7_m2_generic_"))

    def test_invalid_shape_rejected(self):
        with self.assertRaises(ValueError):
            relation_conditions(torch.randn(2,4),torch.randn(1,4))


if __name__ == "__main__":
    unittest.main()
