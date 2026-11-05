"""CPU-only read-only M6.1 expert intervention tests."""
import unittest
import numpy as np
import pandas as pd
import torch

from scripts.audit_k7_m61_frozen_experts import (
    aggregate,predict_interventions,same_image_task_test,validate_partition)
from urban_transfer.k7_m61_router import StructuredMLLMRouter
from urban_transfer.k7_mllm_hidden import TASKS


class M61FrozenAuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=StructuredMLLMRouter(
            12,32,hidden=32,group_counts=(6,4,5)).eval()
        self.x=torch.randn(6,12)
        self.ctx=torch.randn(6,4)
        self.tokens=torch.randn(6,15,32)

    def test_learned_exactly_matches_saved_forward(self):
        with torch.no_grad():
            preds,weights,_=predict_interventions(
                self.model,self.x,self.ctx,self.tokens)
            base,w,_=self.model(self.x,self.ctx,self.tokens,return_aux=True)
        self.assertTrue(torch.allclose(preds["learned"],base,atol=1e-6))
        self.assertTrue(torch.allclose(weights,w,atol=1e-6))

    def test_expert_only_matches_mask_and_equal_is_between_features(self):
        with torch.no_grad():
            pred,_,_=predict_interventions(
                self.model,self.x,self.ctx,self.tokens)
            for k,name in enumerate(("population_only","food_only")):
                mask=torch.zeros((6,2),dtype=torch.bool)
                mask[:,k]=True
                forced=self.model(self.x,self.ctx,self.tokens,mask=mask)
                self.assertTrue(torch.allclose(pred[name],forced,atol=1e-6))
            self.assertEqual(tuple(pred["equal_mix"].shape),(6,))

    def test_same_image_comparison_has_no_training(self):
        cache={(i,t):torch.randn(15,32) for i in range(6) for t in TASKS}
        before={k:v.detach().clone() for k,v in self.model.state_dict().items()}
        report=same_image_task_test(
            self.model,self.x,self.ctx,np.arange(6),cache,3,"cpu")
        self.assertEqual(len(report["per_indicator_population_weight"]),3)
        self.assertEqual(len(report["same_image_pairwise"]),3)
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k])
                            for k,v in before.items()))

    def test_validation_rejects_target_training_leak(self):
        frame=pd.DataFrame({
            "indicator_column":["population_count","education_accessibility",
                                "education_accessibility"],
            "city":["A","A","B"]})
        manifest={"train":[0],"SCUI":[1],"UCUI":[2],"held_cities":["B"]}
        train,scui,ucui=validate_partition(frame,manifest)
        self.assertEqual(len(train),1)
        manifest["train"]=[1]
        manifest["SCUI"]=[0]
        with self.assertRaises(ValueError):
            validate_partition(frame,manifest)

    def test_nonfinite_statistics_fail(self):
        with self.assertRaises(ValueError):
            aggregate([np.nan])


if __name__=="__main__":
    unittest.main()
