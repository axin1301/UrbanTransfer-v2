"""M1 relation intervention tests (no trained checkpoint required)."""
import unittest
import numpy as np
import torch

from scripts.audit_k7_m1_relations import interventions, evaluate_arrays
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS, HELDOUT_TASK, all_relation_texts


class M1RelationAuditTests(unittest.TestCase):
    def test_swap_is_invariant_but_zero_and_substitution_differ(self):
        torch.manual_seed(101)
        tasks = (*TRAIN_TASKS, HELDOUT_TASK)
        semantics = {t: torch.randn(16) for t in tasks}
        cards = {edge: torch.randn(16) for edge in all_relation_texts()}
        options = interventions(semantics, cards)
        self.assertTrue(torch.allclose(options["correct"], options["swap_edges"]))
        self.assertEqual(int(torch.count_nonzero(options["zero_relation"])), 0)
        self.assertFalse(torch.allclose(options["correct"], options["source_training_relation"]))

    def test_no_change_has_zero_deltas(self):
        truth = np.array([0.4, 1.2, 0.9])
        pred = np.array([0.3, 1.0, 0.8])
        m = evaluate_arrays(pred, pred, truth, 0.4, 1.8)
        self.assertAlmostEqual(m["mean_abs_prediction_change"], 0.0)
        self.assertAlmostEqual(m["delta_mse_vs_correct"], 0.0)


if __name__ == "__main__":
    unittest.main()
