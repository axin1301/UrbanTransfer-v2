"""No-data unit tests for K7 hypernetwork, relations and regressors."""
import unittest

import torch
import torch.nn.functional as F

from urban_transfer.k7_hyper import K7HyperRegressor
from urban_transfer.protocol import SOURCE_TASKS
from urban_transfer.z1_relation import fold_relations
from scripts.run_k7_pilot import VARIANTS, diagnostic


class K7Tests(unittest.TestCase):
    def test_variants_are_controlled(self):
        self.assertEqual(VARIANTS, ("k7_e1", "k7_e2", "k7_e2_shuffle"))

    def test_generated_adapter_conditioning_and_gradients(self):
        torch.manual_seed(17)
        model = K7HyperRegressor(12, 1024, hidden=32, rank=4,
                                 use_relation=False)
        image = torch.randn(5, 12)
        context = torch.randn(5, 4)
        semantics = torch.randn(5, 1024)
        out = model(image, context, semantics)
        self.assertEqual(tuple(out.shape), (5,))
        self.assertTrue(torch.isfinite(out).all())
        self.assertFalse(torch.allclose(
            out, model(image, context, semantics.roll(1, 0))))
        F.mse_loss(out, torch.randn(5)).backward()
        self.assertIsNotNone(model.generator[-1].weight.grad)
        self.assertTrue(torch.isfinite(model.generator[-1].weight.grad).all())
        self.assertGreater(float(model.generator[-1].weight.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            model(image, context, semantics, semantics)

    def test_relation_conditioned_adapter_uses_message(self):
        torch.manual_seed(18)
        model = K7HyperRegressor(12, 1024, hidden=32, rank=4,
                                 use_relation=True)
        image = torch.randn(5, 12)
        context = torch.randn(5, 4)
        semantics = torch.randn(5, 1024)
        relations = torch.randn(5, 1024)
        normal = model(image, context, semantics, relations)
        other = model(image, context, semantics, relations.roll(1, 0))
        self.assertFalse(torch.allclose(normal, other))
        normal.square().mean().backward()
        self.assertGreater(float(model.relation[1].weight.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            model(image, context, semantics)

    def test_relation_shuffle_changes_source_binding(self):
        tasks = sorted(SOURCE_TASKS)
        width = 128
        torch.manual_seed(11)
        semantic = {task: F.normalize(torch.randn(width), dim=0)
                    for task in tasks}
        pairs = {(a, b): F.normalize(torch.randn(width), dim=0)
                 for a in tasks for b in tasks if a != b}
        held = tasks[0]
        normal = fold_relations(pairs, held, tasks, semantic)
        shuffle = fold_relations(pairs, held, tasks, semantic, shuffle=True)
        self.assertFalse(torch.allclose(normal[held], shuffle[held]))

    def test_oracle_is_only_diagnostic(self):
        import numpy as np
        record = diagnostic(np.array([1.0, 2.0, 3.0]),
                            np.array([4.0, 5.0, 6.0]))
        self.assertIn("affine_fitted_mse_ORACLE_NOT_ZERO_SHOT", record)


if __name__ == "__main__":
    unittest.main()
