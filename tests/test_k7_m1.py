"""K7 M1: four grounded directed cards, and only two supervised indicators."""
import unittest

from urban_transfer.k7_m1_knowledge import (
    TRAIN_TASKS, HELDOUT_TASK, all_relation_texts, relation_text)
from scripts.run_k7_m1 import semantic_inputs
import torch


class MechanismCardTests(unittest.TestCase):
    def test_exact_four_directed_cards(self):
        pairs = set(all_relation_texts())
        self.assertEqual(pairs, {
            ("population_count", "food_accessibility"),
            ("food_accessibility", "population_count"),
            ("population_count", "education_accessibility"),
            ("food_accessibility", "education_accessibility"),
        })
        self.assertEqual(len(TRAIN_TASKS), 2)
        self.assertNotIn(HELDOUT_TASK, TRAIN_TASKS)

    def test_explicit_uncertainty_and_no_target_moments(self):
        for card in all_relation_texts().values():
            self.assertIn("Do NOT transfer:", card)
            self.assertIn("HYPOTHESIS", card)
            self.assertIn("target means", card)

    def test_validation_relation_uses_only_training_sources(self):
        embeddings = {t: torch.randn(12) for t in (*TRAIN_TASKS, HELDOUT_TASK)}
        relations = {k: torch.randn(12) for k in all_relation_texts()}
        sem, rel = semantic_inputs(HELDOUT_TASK, embeddings, relations)
        self.assertEqual(tuple(sem.shape), (12,))
        self.assertTrue(torch.isfinite(rel).all())
        self.assertEqual(len([s for s,t in relations if t == HELDOUT_TASK]), 2)
        with self.assertRaises(ValueError):
            relation_text(HELDOUT_TASK, TRAIN_TASKS[0])


if __name__ == "__main__":
    unittest.main()
