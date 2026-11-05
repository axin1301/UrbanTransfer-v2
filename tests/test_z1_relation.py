"""Z1 relation / unattended run tests: no Qwen or RemoteCLIP model needed."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.run_z1_relation import VARIANTS, diagnostic
from scripts.run_z1_overnight import collect, summarize
from urban_transfer.protocol import SOURCE_TASKS
from urban_transfer.z1_relation import (
    RelationRegressor, fold_relations, relation_texts)


class Z1RelationTests(unittest.TestCase):
    def test_relation_texts_cover_exact_source_ordered_pairs(self):
        texts = relation_texts()
        self.assertEqual(len(texts), len(SOURCE_TASKS)*(len(SOURCE_TASKS)-1))
        self.assertFalse(any(a == b for a, b in texts))
        self.assertTrue(all("unverified" in t.lower() for t in texts.values()))

    def test_relation_pair_binding_and_permutation_are_nontrivial(self):
        tasks = sorted(SOURCE_TASKS)
        held = tasks[0]
        width = len(tasks)*len(tasks)
        cache = {}
        for i, a in enumerate(tasks):
            for j, b in enumerate(tasks):
                if a != b:
                    vec = torch.zeros(width)
                    vec[i*len(tasks)+j] = 1.
                    cache[a, b] = vec
        identity = {name: torch.randn(width).abs().add(0.1)
                    for name in tasks}
        normal = fold_relations(cache, held, tasks, identity)
        shuffled = fold_relations(cache, held, tasks, identity, shuffle=True)
        self.assertEqual(set(normal), set(tasks))
        self.assertEqual(set(shuffled), set(tasks))
        self.assertTrue(any(not torch.allclose(normal[t], shuffled[t])
                            for t in tasks))
        for target in tasks:
            for source in tasks:
                if source == held or source == target:
                    index = tasks.index(target)*len(tasks)+tasks.index(source)
                    self.assertEqual(float(normal[target][index]), 0.)


    def test_relation_model_grads_and_conditioning(self):
        torch.manual_seed(1)
        model = RelationRegressor(12, 16, mode="b3", hidden=32, rank=8)
        images = torch.randn(5, 12)
        context = torch.randn(5, 4)
        sem = torch.randn(5, 16)
        rel = torch.randn(5, 16)
        out = model(images, context, sem, rel)
        self.assertEqual(tuple(out.shape), (5,))
        self.assertTrue(torch.isfinite(out).all())
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(model.project[1].weight.grad).all())
        other = model(images, context, sem, rel.flip(0))
        self.assertFalse(torch.allclose(out, other))

    def test_oracle_diagnostic_marked_as_oracle(self):
        d = diagnostic(np.array([2., 4., 6.]), np.array([1., 3., 5.]))
        self.assertIn("ORACLE_NOT_ZERO_SHOT", next(k for k in d if "ORACLE" in k))
        self.assertAlmostEqual(d["prediction_bias_oracle"], 1.0)
        self.assertAlmostEqual(d["affine_fitted_mse_ORACLE_NOT_ZERO_SHOT"], 0.0)

    def test_sweep_collects_only_completed_macro_rows(self):
        self.assertIn("relation_shuffle_b3", VARIANTS)
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            sub = folder/"short_b1_seed42"
            sub.mkdir()
            with (sub/"macro_metrics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "variant", "seed", "partition", "mean_mse_log1p",
                    "mean_pearson", "mean_relative_mse"])
                writer.writeheader()
                writer.writerow({"variant": "short_b1", "seed": 42,
                                 "partition": "UCUI", "mean_mse_log1p": 1.2,
                                 "mean_pearson": .4, "mean_relative_mse": .9})
            from types import SimpleNamespace
            a = SimpleNamespace(output=folder, variants=["short_b1"])
            jobs = [("short_b1", 42, sub)]
            rows = collect(a, jobs)
            self.assertEqual(len(rows), 1)
            summarize(a, jobs, [{"variant": "short_b1", "seed": 42,
                                  "state": "OK", "log": str(sub/"console.log")}])
            self.assertTrue((folder/"REPORT.md").exists())
            self.assertIn("1.200", (folder/"REPORT.md").read_text())


if __name__ == "__main__":
    unittest.main()
