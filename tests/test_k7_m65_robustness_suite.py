"""CPU tests for frozen M6.5 robustness experiment: source-only statistics."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.run_k7_m65_robustness_suite import (
    city_mean_neighborhood,stat,summarize,
    save_prediction,read_prediction,bootstrap_city_effect,rows_three_decimals)
from scripts.run_k7_m65_dev_intervention import permute_physical_neighborhood
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        records=[]
        for city in ("A","B","C"):
            for tile in ("t1","t2","t3"):
                for task in TRAIN_TASKS:
                    records.append({
                        "city":city,"image_name":tile,"indicator_column":task,
                        "indicator_value_log1p":float(len(records))})
        self.frame=pd.DataFrame(records)
        self.ids=np.arange(len(records),dtype=np.int64)
        self.nb=torch.arange(
            len(records)*5,dtype=torch.float32).reshape(-1,5)
        for i in range(0,len(records),2):
            self.nb[i+1]=self.nb[i]

    def test_city_mean_equal_within_city_and_label_independent(self):
        actual=city_mean_neighborhood(self.frame,self.nb,self.ids)
        for i in range(0,len(self.ids),6):
            self.assertTrue(torch.equal(actual[i],actual[i+5]))
        modified=self.frame.copy()
        modified["indicator_value_log1p"]=1e8
        modified["indicator_column"]="not_a_source_task"
        changed=city_mean_neighborhood(modified,self.nb,self.ids)
        self.assertTrue(torch.equal(actual,changed))

    def test_zero_and_permuted_all_label_free(self):
        changed=permute_physical_neighborhood(
            self.frame,self.nb,self.ids,seed=10001)
        modified=self.frame.copy()
        modified["indicator_column"]="unknown"
        modified["indicator_value_log1p"]=-999
        changed_again=permute_physical_neighborhood(
            modified,self.nb,self.ids,seed=10001)
        self.assertTrue(torch.equal(changed,changed_again))
        self.assertTrue(bool((changed!=self.nb).any()))

    def test_predictions_atomic_roundtrip_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"perm_000.npz"
            predictions=np.arange(len(self.ids),dtype=np.float64)*.1
            routes=np.arange(len(self.ids),dtype=np.float64)*.005
            save_prediction(path,self.ids,predictions,routes)
            restored,weights=read_prediction(path,self.ids)
            np.testing.assert_allclose(restored,predictions,atol=0,rtol=0)
            np.testing.assert_allclose(weights,routes,rtol=1e-6)
            with self.assertRaisesRegex(ValueError,"indices"):
                read_prediction(path,self.ids[::-1])

    def test_summarize_direction_and_city_bootstrap(self):
        truth=np.arange(len(self.ids),dtype=float)*.01
        real=truth+1.
        worse=truth+2.
        pred={
            "real":(real,np.ones(len(truth))),
            "zero":(worse,np.ones(len(truth))),
            "city_mean":(worse,np.ones(len(truth))),
            "perm_000":(worse,np.ones(len(truth))),
            "perm_001":(worse,np.ones(len(truth))),
        }
        metrics,cities,summary=summarize(
            self.frame,self.ids,pred,truth,2)
        self.assertEqual(len(metrics),len(TRAIN_TASKS)*5)
        self.assertEqual(len(cities),len(TRAIN_TASKS)*4*3)
        record=next(x for x in summary if x["task"]==TRAIN_TASKS[0]
                    and x["comparison"]=="in_city_tile_derangement")
        self.assertAlmostEqual(record["perturbed_minus_real_mse_mean"],3.)
        self.assertAlmostEqual(record["permutation_p_one_sided"],1/3)
        self.assertEqual(record["fraction_held_cities_worse"],1.)
        self.assertAlmostEqual(record["city_bootstrap_ci_low"],3.)
        self.assertAlmostEqual(record["city_bootstrap_ci_high"],3.)
        self.assertEqual(rows_three_decimals([{"v":1.23456}]),
                         [{"v":"1.235"}])

    def test_bootstrap_does_not_resample_rows(self):
        vals=[{"city":"A","n_rows":100,"perturbed_minus_real":1.},
              {"city":"B","n_rows":1,"perturbed_minus_real":-1.}]
        lo,hi=bootstrap_city_effect(vals,seed=33,reps=500)
        self.assertLessEqual(lo,-.9)
        self.assertGreaterEqual(hi,.9)


if __name__=="__main__":
    unittest.main()
