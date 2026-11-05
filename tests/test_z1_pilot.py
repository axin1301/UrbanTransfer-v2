"""Z1 safeguards and architecture sanity checks; no images or Qwen GPU needed."""
import unittest

import numpy as np
import pandas as pd
import torch

from urban_transfer.z1_model import Z1Regressor
from scripts.run_z1_pilot import fold_indices, calibrated_moments


class Z1Tests(unittest.TestCase):
    def test_source_city_and_task_disjointness(self):
        frame = pd.DataFrame([
            {"city": city, "indicator_column": task,
             "indicator_value_log1p": float(i)}
            for city in ("a", "b", "c", "d")
            for i, task in enumerate(("population_count", "food_accessibility"))])
        splits = fold_indices(frame, "population_count", {"d"})
        all_ids = np.concatenate(list(splits.values())).tolist()
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), set(range(len(frame))))
        train = frame.iloc[splits["train"]]
        self.assertFalse(train.indicator_column.eq("population_count").any())
        self.assertFalse(train.city.eq("d").any())
        self.assertTrue(frame.iloc[splits["UCUI"]].city.eq("d").all())
        self.assertTrue(frame.iloc[splits["UCUI"]].indicator_column.eq("population_count").all())

    def test_calibration_excludes_heldout_target(self):
        frame = pd.DataFrame({
            "indicator_column": ["population_count"] * 3 + ["food_accessibility"] * 3,
            "indicator_value_log1p": [1., 2., 3., 4., 5., 6.]})
        semantics = {
            "population_count": torch.tensor([1., 0.]),
            "food_accessibility": torch.tensor([0., 1.]),
            "healthcare_accessibility": torch.tensor([0.8, 0.2])}
        stats, unseen = calibrated_moments(frame, semantics, "healthcare_accessibility")
        self.assertEqual(set(stats), {"population_count", "food_accessibility"})
        self.assertTrue(np.isfinite(unseen).all())
        poisoned = pd.concat([frame, pd.DataFrame({
            "indicator_column": ["healthcare_accessibility"],
            "indicator_value_log1p": [9999.]})], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "leaked"):
            calibrated_moments(poisoned, semantics, "healthcare_accessibility")

    def test_models_have_gradients_and_respond_to_task_semantics(self):
        torch.manual_seed(0)
        images = torch.randn(7, 12)
        contexts = torch.randn(7, 4)
        semantics = torch.randn(7, 10)
        for mode in ("b1", "b2", "b3"):
            with self.subTest(mode=mode):
                model = Z1Regressor(12, 10, mode=mode, hidden=32, rank=8)
                output = model(images, contexts, semantics)
                self.assertEqual(tuple(output.shape), (7,))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(any(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters()))
                swapped = model(images, contexts, semantics.flip(0))
                self.assertFalse(torch.allclose(output, swapped))

    def test_invalid_mode_and_empty_split(self):
        with self.assertRaises(ValueError):
            Z1Regressor(10, 10, "unknown")
        frame = pd.DataFrame({"city": ["a", "b"],
                              "indicator_column": ["population_count"] * 2})
        with self.assertRaisesRegex(ValueError, "Empty"):
            fold_indices(frame, "population_count", {"b"})


if __name__ == "__main__":
    unittest.main()
