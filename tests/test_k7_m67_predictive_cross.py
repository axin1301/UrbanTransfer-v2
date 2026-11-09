"""M6.7 CPU tests for OOF source-city safety and predictive context."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from urban_transfer.k7_m1_knowledge import TRAIN_TASKS
from urban_transfer.k7_m67_oof_indicator_context import (
    crossfit_source_indicator_predictions,load_crossfit_context)
from urban_transfer.k7_m67_predictive_neighbor import PredictiveNeighborResidual
from scripts.run_k7_m67_predictive_cross_dev import arm_predicted_features


class M67Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        rows=[]
        for city in ("A","B","C","D","E","F"):
            for tile in range(5):
                for task in TRAIN_TASKS:
                    rows.append({"city":city,"image_name":f"{tile}.png",
                                 "indicator_column":task,
                                 "patch_lon":tile*.01,
                                 "patch_lat":55.})
        self.frame=pd.DataFrame(rows)
        self.x=torch.from_numpy(rng.standard_normal((len(rows),12)).astype("float32"))
        for i in range(0,len(rows),2):
            self.x[i+1]=self.x[i]
        self.geo=torch.from_numpy(rng.standard_normal((len(rows),4)).astype("float32"))
        for i in range(0,len(rows),2):
            self.geo[i+1]=self.geo[i]
        self.y=torch.from_numpy(rng.standard_normal(len(rows)).astype("float32"))
        self.names=self.frame.indicator_column.tolist()
        self.train=np.flatnonzero(~self.frame.city.isin(["F"]).to_numpy())
        self.stats={task:(0.,1.) for task in TRAIN_TASKS}

    def test_city_fold_predictions_are_label_isolated(self):
        base=crossfit_source_indicator_predictions(
            self.frame,self.x,self.geo,self.y,self.names,
            self.train,self.stats)
        changed=self.y.clone()
        mask=self.frame.city.eq("A").to_numpy()
        changed[mask]=1e5
        different=crossfit_source_indicator_predictions(
            self.frame,self.x,self.geo,changed,self.names,
            self.train,self.stats)
        np.testing.assert_allclose(
            base[mask].numpy(),different[mask].numpy(),atol=1e-5,rtol=0)
        self.assertTrue(torch.isfinite(base).all())
        self.assertEqual(tuple(base.shape),(len(self.frame),2))

    def test_task_missing_in_one_train_city_is_filled_without_leakage(self):
        # Source-fit city E has only Population training labels; Food
        # nevertheless needs predicted values for EVERY physical tile.
        source_train=np.asarray([
            i for i in self.train
            if not (self.frame.city.iloc[int(i)]=="E" and
                    self.names[int(i)]=="food_accessibility")
        ],dtype=np.int64)
        base=crossfit_source_indicator_predictions(
            self.frame,self.x,self.geo,self.y,self.names,
            source_train,self.stats)
        changed=self.y.clone()
        city_e=self.frame.city.eq("E").to_numpy()
        changed[city_e]=100000.
        again=crossfit_source_indicator_predictions(
            self.frame,self.x,self.geo,changed,self.names,
            source_train,self.stats)
        food_ix=TRAIN_TASKS.index("food_accessibility")
        # E has no Food training labels: its Food predictions come from
        # full Food fit on the other source-fit cities and remain fixed.
        np.testing.assert_allclose(
            base[city_e,food_ix].numpy(),
            again[city_e,food_ix].numpy(),rtol=0,atol=0)
        # Source-fit city E still has Population labels, so its
        # Population predictions are OOF (also city-label isolated).
        pop_ix=TRAIN_TASKS.index("population_count")
        np.testing.assert_allclose(
            base[city_e,pop_ix].numpy(),
            again[city_e,pop_ix].numpy(),rtol=0,atol=1e-5)
        self.assertTrue(bool(torch.isfinite(base).all()))
        self.assertEqual(tuple(base.shape),(len(self.frame),2))

    def test_held_city_labels_do_not_affect_context(self):
        base=crossfit_source_indicator_predictions(
            self.frame,self.x,self.geo,self.y,self.names,
            self.train,self.stats)
        altered=self.y.clone()
        altered[self.frame.city.eq("F").to_numpy()]=1e10
        same=crossfit_source_indicator_predictions(
            self.frame,self.x,self.geo,altered,self.names,
            self.train,self.stats)
        self.assertTrue(torch.equal(base,same))

    def test_arm_inputs_and_shapes(self):
        from urban_transfer.k7_m66_indicator_graph import physical_neighbor_graph
        graph=physical_neighbor_graph(self.frame,k=3)
        vals=torch.randn(len(self.frame),2)
        ids=np.arange(6,dtype=np.int64)
        cross,dist,mask=arm_predicted_features(
            vals,graph,ids,self.names,"cross_indicator")
        same,_,_=arm_predicted_features(
            vals,graph,ids,self.names,"same_indicator")
        zero,_,_=arm_predicted_features(
            vals,graph,ids,self.names,"no_predictive")
        self.assertEqual(tuple(cross.shape),(6,3,2))
        self.assertTrue(torch.equal(same[:,:,0],same[:,:,1]))
        self.assertEqual(int(torch.count_nonzero(zero)),0)
        self.assertTrue(bool(mask.any(dim=1).all()))
        self.assertTrue(bool((cross[:,:,0]!=cross[:,:,1]).any()))

    def test_identical_model_init_and_backprop(self):
        torch.manual_seed(42)
        model=PredictiveNeighborResidual(
            12,24,hidden=32,heads=4,group_counts=(6,4,5))
        x=self.x[:4]; geo=self.geo[:4]
        t=torch.randn(4,15,24)
        nb=torch.randn(4,38)
        p=torch.randn(4,3,2)
        d=torch.zeros(4,3)
        mask=torch.ones(4,3,dtype=torch.bool)
        pred,w,task,fused,base,delta=model(
            x,geo,t,nb,p,d,mask,return_aux=True)
        self.assertTrue(torch.equal(pred,base))
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))
        loss=(pred.square().mean()+
              model.source_expert_predictions(x,geo,t,nb,p,d,mask).square().mean()+
              model.source_task_separation(
                  x,geo,t,t+torch.randn_like(t),nb,p,d,mask)+
              model.target_image_alignment(x,geo,t,nb,p,d,mask))
        loss.backward()
        self.assertIsNotNone(model.predicted_encoder[1].weight.grad)
        self.assertTrue(all(not param.requires_grad
                            for param in model.base.parameters()))

    def test_metadata_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=Path(folder)/"manifest.json"
            manifest.write_text('{"seed":42}')
            path=Path(folder)/"oof.pt"
            a=load_crossfit_context(self.frame,self.x,self.geo,self.y,
                                    self.names,self.train,self.stats,
                                    manifest,path)
            b=load_crossfit_context(self.frame,self.x,self.geo,self.y,
                                    self.names,self.train,self.stats,
                                    manifest,path)
            self.assertTrue(torch.equal(a,b))
            with self.assertRaisesRegex(ValueError,"mismatch"):
                load_crossfit_context(
                    self.frame,self.x,self.geo,self.y*2,
                    self.names,self.train,self.stats,manifest,path)


if __name__=="__main__":
    unittest.main()
