"""CPU tests for M6.2 task-queried visual routing and source-only protocol."""
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from urban_transfer.k7_m62_task_query import TaskQueriedRouter
from urban_transfer.k7_mllm_hidden import TASKS
from scripts.run_k7_m62_task_query import (
    validate_split,frozen_metrics,same_image_prompt_audit)


class TaskQueriedM62Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=TaskQueriedRouter(
            12,24,hidden=32,heads=4,group_counts=(6,4,5))
        self.visual=torch.randn(8,12)
        self.context=torch.randn(8,4)
        self.source=torch.randn(8,15,24)
        self.target=self.source.clone()
        self.target[:,6:]+=torch.randn(8,9,24)*2

    def test_same_image_text_change_affects_task_query_and_router(self):
        self.model.eval()
        with torch.no_grad():
            a=self.model.condition(self.source)
            b=self.model.condition(self.target)
            _,wa,qa,_=self.model(self.visual,self.context,self.source,
                                 return_aux=True)
            _,wb,qb,_=self.model(self.visual,self.context,self.target,
                                 return_aux=True)
        self.assertGreater(float((qa-qb).abs().mean()),1e-5)
        self.assertGreater(float((wa-wb).abs().mean()),1e-8)
        self.assertTrue(torch.allclose(qa,a[1]))
        self.assertTrue(torch.allclose(qb,b[1]))

    def test_image_query_cross_attention_gradient_and_source_experts(self):
        target_values=torch.randn(8)
        loss_main,pop_weights,_,_=self.model(
            self.visual,self.context,self.source,return_aux=True)
        self.assertEqual(tuple(pop_weights.shape),(8,2))
        self.assertTrue(bool((pop_weights>0).all()))
        own=self.model.source_expert_predictions(
            self.visual,self.context,self.source)
        names=torch.tensor([0,1,0,1,0,1,0,1]).unsqueeze(1)
        direct=own.gather(1,names).squeeze(1)
        loss=(F.mse_loss(loss_main,target_values)+
              .3*F.mse_loss(direct,target_values)+
              .2*self.model.source_task_separation(
                  self.source,self.target)+
              .05*self.model.target_image_alignment(
                  self.target,self.visual))
        loss.backward()
        for parameter in (
            self.model.task_query[-1].weight,
            self.model.cross_attention.in_proj_weight,
            self.model.router[-1].weight,
            self.model.experts[0][1].weight,
            self.model.experts[1][1].weight,
            self.model.source_task_head.weight,
            self.model.target_align_head.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()),0.)

    def test_source_classification_is_same_image_multitask(self):
        loss=self.model.source_task_separation(self.source,self.target)
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):
            self.model.source_task_separation(self.source,self.target[:4])

    def test_frozen_metrics_and_prompts_do_not_update_checkpoint(self):
        self.model.eval()
        labels=torch.randn(8)
        cache={(i,TASKS[2]):self.target[i] for i in range(8)}
        for i in range(8):
            cache[(i,TASKS[0])]=self.source[i]
            cache[(i,TASKS[1])]=self.target[i]
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        report=frozen_metrics(
            self.model,self.visual,self.context,labels,
            np.arange(8,dtype=np.int64),cache,0.,1.,4,"cpu")
        self.assertEqual(report["n_rows"],8)
        self.assertIn("mismatch_change_log1p",report)
        same=same_image_prompt_audit(
            self.model,self.visual,self.context,
            np.arange(8,dtype=np.int64),cache,4,"cpu")
        self.assertEqual(len(same),3)
        self.assertTrue(all(torch.equal(a,self.model.state_dict()[k])
                            for k,a in before.items()))

    def test_strict_city_and_indicator_holdout(self):
        df=pd.DataFrame({"city":["A","A","B"],
                         "indicator_column":[TASKS[0],TASKS[2],TASKS[2]]})
        manifest={"train":[0],"SCUI":[1],"UCUI":[2],
                  "held_cities":["B"]}
        self.assertEqual(len(validate_split(df,manifest)[0]),1)
        manifest["train"]=[1]
        manifest["SCUI"]=[0]
        with self.assertRaises(ValueError):
            validate_split(df,manifest)


if __name__=="__main__":
    unittest.main()
