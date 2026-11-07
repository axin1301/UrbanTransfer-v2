"""M6.5 CPU synthetic tests: no label leakage, per-city tiles, bounded model."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.run_k7_m65_full import (
    city_dev_split,fit_source_moments,evaluate_source_or_target)
from urban_transfer.k7_m65_neighborhood import (
    city_neighborhood_features,load_or_build_neighborhood)
from urban_transfer.k7_m65_neighborhood_student import (
    NeighborhoodCityIndicatorResidual)
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK


class M65Tests(unittest.TestCase):
    def setUp(self):
        rows=[]
        for city in ("A","B","C","D","E","F","G","H"):
            for tile in range(4):
                for task in (*TRAIN_TASKS,HELDOUT_TASK):
                    rows.append({
                        "city":city,"image_name":f"{tile}.png",
                        "indicator_column":task,
                        "patch_lon":float(tile)+ord(city)*.01,
                        "patch_lat":55.+ord(city)*.001,
                        "indicator_value_log1p":float(tile)})
        self.frame=pd.DataFrame(rows)
        self.visual=torch.randn(len(rows),12)
        # Same physical tile under all 3 tasks MUST share RemoteCLIP.
        for i in range(0,len(rows),3):
            self.visual[i+1:i+3]=self.visual[i]
        self.ctx=torch.randn(len(rows),4)

    def test_label_free_neighbor_invariant(self):
        a=city_neighborhood_features(self.frame,self.visual,k=8)
        changed=self.frame.copy()
        changed["indicator_value_log1p"]=1e10
        changed["indicator_column"]="different-task"
        b=city_neighborhood_features(changed,self.visual,k=8)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(tuple(a.shape),(len(self.frame),38))
        for i in range(0,len(self.frame),3):
            self.assertTrue(torch.equal(a[i],a[i+1]))
            self.assertTrue(torch.equal(a[i],a[i+2]))

    def test_city_neighborhood_isolated(self):
        a=city_neighborhood_features(self.frame,self.visual,k=3)
        changed=self.visual.clone()
        changed[self.frame.city.eq("B").to_numpy()]+=9999
        b=city_neighborhood_features(self.frame,changed,k=3)
        rows=self.frame.city.eq("A").to_numpy()
        self.assertTrue(torch.equal(a[rows],b[rows]))

    def test_city_holdout_is_strict_source_only(self):
        task=self.frame.indicator_column.to_numpy()
        source_ids=np.flatnonzero(np.isin(task,TRAIN_TASKS))
        fit,held,cities=city_dev_split(self.frame,source_ids)
        actual=self.frame.city.astype(str).to_numpy()
        self.assertFalse(set(actual[fit])&set(actual[held]))
        self.assertEqual(set(actual[held]),set(cities))
        self.assertTrue(set(task[np.r_[fit,held]])<=set(TRAIN_TASKS))
        self.assertEqual(len(fit)+len(held),len(source_ids))
        self.assertEqual(city_dev_split(self.frame,source_ids)[2],cities)

    def test_cache_consistency(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            manifest=path/"manifest.json"
            manifest.write_text('{"seed":42}')
            target=path/"neighbors.pt"
            a=load_or_build_neighborhood(
                self.frame,self.visual,manifest,target,k=3)
            b=load_or_build_neighborhood(
                self.frame,self.visual,manifest,target,k=3)
            self.assertTrue(torch.equal(a,b))
            with self.assertRaises(ValueError):
                load_or_build_neighborhood(
                    self.frame,self.visual,manifest,target,k=8)

    def test_student_initial_equals_visual_and_city_condition_matters(self):
        torch.manual_seed(42)
        model=NeighborhoodCityIndicatorResidual(
            12,24,hidden=32,heads=4,group_counts=(6,4,5))
        x=self.visual[:6]
        geo=self.ctx[:6]
        nb=city_neighborhood_features(self.frame,self.visual,k=3)[:6]
        tokens=torch.randn(6,15,24)
        with torch.no_grad():
            pred,weight,task,fused,base,delta=model(
                x,geo,tokens,nb,return_aux=True)
        self.assertTrue(torch.equal(pred,base))
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))
        self.assertEqual(tuple(weight.shape),(6,2))
        # LayerNorm is invariant to adding the same constant to every
        # channel: nb+2 is NOT a valid neighborhood-sensitivity test.
        # Perturb the spatial feature pattern, not its global offset.
        altered=nb.clone()
        altered[:,:12]=-altered[:,:12]
        with torch.no_grad():
            city_a=model.condition(x,geo,tokens,nb)[1]
            city_b=model.condition(x,geo,tokens,altered)[1]
        self.assertGreater(float((city_a-city_b).abs().mean()),1e-6)

    def test_student_source_training_backward(self):
        torch.manual_seed(42)
        model=NeighborhoodCityIndicatorResidual(
            12,24,hidden=32,heads=4,group_counts=(6,4,5))
        neighborhood=city_neighborhood_features(self.frame,self.visual,k=3)
        x=self.visual[:6]
        ctx=self.ctx[:6]
        nb=neighborhood[:6]
        h=torch.randn(6,15,24)
        pop=torch.randn(6,15,24)
        food=torch.randn(6,15,24)
        edu=torch.randn(6,15,24)
        before={key:value.clone() for key,value
                in model.base.state_dict().items()}
        prediction=model(x,ctx,h,nb)
        own=model.source_expert_predictions(x,ctx,h,nb)
        loss=(prediction.square().mean()+own.square().mean()+
              model.source_task_separation(x,ctx,pop,food,nb)+
              model.target_image_alignment(x,ctx,edu,nb))
        loss.backward()
        self.assertTrue(any(p.grad is not None
                            for p in model.neighborhood_encoder.parameters()))
        self.assertTrue(all(not p.requires_grad
                            for p in model.base.parameters()))
        for k,v in before.items():
            self.assertTrue(torch.equal(v,model.base.state_dict()[k]))


if __name__=="__main__":
    unittest.main()
