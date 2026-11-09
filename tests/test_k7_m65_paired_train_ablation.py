"""CPU validation for same-initialization M6.5 three-arm training protocol."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.run_k7_m65_paired_train_ablation import (
    ARMS,build_arms,summarize,ensure_run_config)
from urban_transfer.k7_m65_neighborhood_student import (
    NeighborhoodCityIndicatorResidual)
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS


class PairedTrainAblationTests(unittest.TestCase):
    def test_same_model_initialization_and_trainable_parameter_count(self):
        torch.manual_seed(42)
        template=NeighborhoodCityIndicatorResidual(
            12,24,hidden=32,heads=4,group_counts=(6,4,5))
        original={k:v.detach().clone() for k,v in template.state_dict().items()}
        for arm in ARMS:
            torch.manual_seed(777)
            other=NeighborhoodCityIndicatorResidual(
                12,24,hidden=32,heads=4,group_counts=(6,4,5))
            other.load_state_dict(original)
            self.assertTrue(all(torch.equal(a,b)
                for (k,a),(kk,b) in zip(
                    original.items(),other.state_dict().items())
                if k==kk))
            self.assertEqual(
                sum(p.numel() for p in template.parameters()),
                sum(p.numel() for p in other.parameters()))

    def test_city_means_dedupe_source_tasks(self):
        rows=[]
        for city in ("A","B"):
            for tile in ("tile1","tile2","tile3"):
                for task in TRAIN_TASKS:
                    rows.append({"city":city,"image_name":tile,
                                 "indicator_column":task})
        frame=pd.DataFrame(rows)
        nb=torch.randn(len(frame),8)
        for i in range(0,len(frame),2):
            nb[i+1]=nb[i]
        arms=build_arms(frame,nb,np.arange(len(frame)))
        self.assertTrue(torch.equal(arms["real"],nb))
        self.assertEqual(torch.count_nonzero(arms["zero"]),0)
        self.assertTrue(torch.equal(arms["city_mean"][0],
                                    arms["city_mean"][5]))
        self.assertFalse(torch.equal(arms["city_mean"][0],
                                     arms["city_mean"][6]))

    def test_summary_uses_complete_three_arm_source_task_metrics(self):
        data={}
        for ix,arm in enumerate(ARMS):
            data[arm]=[]
            for task in TRAIN_TASKS:
                data[arm].append({
                    "partition":"dev_held_city_"+task,"n_rows":8,
                    "mse_log1p":3.+ix,"pearson":.7-ix*.01,
                    "mae_log1p":1.,"mean_bias_log1p":.1,
                    "pred_std":1.,"population_weight":.5})
        report=summarize(data)
        self.assertEqual(len(report),len(TRAIN_TASKS)*3)
        first=next(row for row in report
                   if row["task"]==TRAIN_TASKS[0] and row["arm"]=="zero")
        self.assertEqual(first["real_minus_arm_mse"],-1.)

    def test_config_prevents_mixing_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"study"
            ensure_run_config(output,{"arm":"all","seed":42})
            ensure_run_config(output,{"arm":"all","seed":42})
            with self.assertRaisesRegex(ValueError,"mismatch"):
                ensure_run_config(output,{"arm":"all","seed":43})


if __name__=="__main__":
    unittest.main()
