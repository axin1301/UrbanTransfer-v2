"""CPU tests for M6.5 source-only zero-neighborhood matched-capacity control."""
import unittest

import numpy as np
import torch

from scripts.run_k7_m65_zero_neighbor_ablation import (
    assert_same_anchor,source_city_comparison)
from urban_transfer.k7_m65_neighborhood_student import (
    NeighborhoodCityIndicatorResidual)


class NoNeighborAblationTests(unittest.TestCase):
    def test_zero_context_train_eval_and_parameter_count(self):
        torch.manual_seed(42)
        model=NeighborhoodCityIndicatorResidual(
            12,24,hidden=32,heads=4,group_counts=(6,4,5))
        x=torch.randn(4,12)
        geo=torch.randn(4,4)
        h=torch.randn(4,15,24)
        # Even with zero neighborhood, whole model architecture is active.
        zeros=torch.zeros(4,model.neighborhood_dim)
        result=model(x,geo,h,zeros)
        self.assertEqual(tuple(result.shape),(4,))
        loss=result.square().mean()+model.source_task_separation(
            x,geo,h,h+torch.randn_like(h),zeros)
        loss.backward()
        self.assertIsNotNone(model.neighborhood_encoder[1].weight.grad)
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in NeighborhoodCityIndicatorResidual(
                12,24,hidden=32,heads=4,group_counts=(6,4,5)).parameters()))

    def test_protected_anchor_is_identical(self):
        model=NeighborhoodCityIndicatorResidual(
            12,24,hidden=32,heads=4,group_counts=(6,4,5))
        original={k:v.detach().clone()
                  for k,v in model.base.state_dict().items()}
        assert_same_anchor(original,model)
        original[next(iter(original))]+=1.
        with self.assertRaisesRegex(ValueError,"frozen"):
            assert_same_anchor(original,model)

    def test_comparison_requires_identical_source_validation_rows(self):
        ref=[{"partition":"dev_held_city_population_count",
              "n_rows":"3","mse_log1p":"3.0","mae_log1p":"1.2",
              "pearson":"0.7","mean_bias_log1p":"-0.2",
              "population_weight":"0.99"}]
        ctl=[{"partition":"dev_held_city_population_count",
              "n_rows":3,"mse_log1p":2.0,"mae_log1p":1.,
              "pearson":.6,"mean_bias_log1p":-.1,
              "population_weight":.5}]
        compared=source_city_comparison(ref,ctl)
        mse=next(x for x in compared if x["metric"]=="mse_log1p")
        self.assertEqual(mse["real_minus_zero"],1.)
        ctl[0]["n_rows"]=4
        with self.assertRaisesRegex(ValueError,"sample mismatch"):
            source_city_comparison(ref,ctl)


if __name__=="__main__":
    unittest.main()
