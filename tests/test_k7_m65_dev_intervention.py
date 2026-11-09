"""CPU invariants for M6.5 frozen held-source-city neighborhood interventions."""
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.run_k7_m65_dev_intervention import (
    permute_physical_neighborhood,compare_rows)


class InterventionTests(unittest.TestCase):
    def setUp(self):
        rows=[]
        for city in ("A","B"):
            for tile in ("t1","t2","t3","t4"):
                for task in ("population_count","food_accessibility"):
                    rows.append({"city":city,"image_name":tile,
                                 "indicator_column":task})
        self.frame=pd.DataFrame(rows)
        self.context=torch.arange(len(rows),dtype=torch.float32)[:,None].repeat(1,5)
        for i in range(0,len(rows),2):
            self.context[i+1]=self.context[i]
        self.held=np.arange(len(rows),dtype=np.int64)

    def test_city_and_tile_consistency(self):
        changed=permute_physical_neighborhood(
            self.frame,self.context,self.held)
        # every physical tile has a single consistent donor
        for i in range(0,len(self.held),2):
            self.assertTrue(torch.equal(changed[i],changed[i+1]))
            self.assertFalse(torch.equal(changed[i],self.context[i]))
            my_city=self.frame.iloc[i].city
            city_ids=self.frame.index[self.frame.city==my_city]
            self.assertTrue(any(
                torch.equal(changed[i],self.context[int(j)])
                for j in city_ids))
        self.assertTrue(
            torch.equal(
                self.context[0:8].unique(dim=0).sort(dim=0).values,
                changed[0:8].unique(dim=0).sort(dim=0).values))
        self.assertTrue(
            torch.equal(
                self.context[8:].unique(dim=0).sort(dim=0).values,
                changed[8:].unique(dim=0).sort(dim=0).values))

    def test_no_read_of_indicator_label_for_permutation(self):
        altered=self.frame.copy()
        altered["indicator_column"]="different"
        altered["indicator_value_log1p"]=1e12
        a=permute_physical_neighborhood(
            self.frame,self.context,self.held)
        b=permute_physical_neighborhood(
            altered,self.context,self.held)
        self.assertTrue(torch.equal(a,b))

    def test_no_cross_city_donor(self):
        changed=permute_physical_neighborhood(
            self.frame,self.context,self.held)
        self.assertTrue(bool((changed[:8,0]<8).all()))
        self.assertTrue(bool((changed[8:,0]>=8).all()))

    def test_requires_two_tiles_per_held_city(self):
        with self.assertRaisesRegex(ValueError,"unique physical"):
            permute_physical_neighborhood(
                self.frame,self.context,np.array([0,1],dtype=np.int64))

    def test_comparison_sign(self):
        data=[]
        for condition,mse in (("real",2.),("zero",3.),("city_permuted",2.5)):
            data.append({"partition":"dev_held_city_population_count",
                         "condition":condition,"mse_log1p":mse,
                         "pearson":.7,"mae_log1p":1.,
                         "mean_bias_log1p":.1,"pred_std":1.,
                         "population_weight":.6})
        comparison=compare_rows(data)
        zero_mse=next(row for row in comparison
                      if row["condition"]=="zero"
                      and row["metric"]=="mse_log1p")
        self.assertEqual(zero_mse["real_minus_perturbed"],-1.)


if __name__=="__main__":
    unittest.main()
