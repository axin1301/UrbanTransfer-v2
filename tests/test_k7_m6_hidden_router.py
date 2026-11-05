"""M6 hidden-state teacher/router and source-only subset CPU tests."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from urban_transfer.k7_mllm_hidden import HiddenStore,task_prompt,TASKS
from urban_transfer.k7_mllm_router import (
    MLLMHiddenRouter,mllm_teacher_distribution,router_distillation)
from scripts.cache_k7_m6_hidden import selection,jobs
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK


class M6Tests(unittest.TestCase):
    def test_prompts_target_relations_and_no_labels(self):
        text=task_prompt(HELDOUT_TASK)
        self.assertIn("population_count",text)
        self.assertIn("food_accessibility",text)
        self.assertIn("education_accessibility",text)
        self.assertNotIn("indicator_value_log1p",text)
        self.assertEqual(len(TASKS),3)

    def test_hidden_sqlite_roundtrip_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=HiddenStore(Path(directory)/"hidden.db")
            self.assertIsNone(cache.get("a"))
            x=torch.randn(8,16)
            cache.put("a","education_accessibility",x)
            cache.put("a","population_count",torch.zeros_like(x))
            y=cache.get("a")
            self.assertEqual(tuple(y.shape),(8,16))
            self.assertTrue(torch.allclose(x,y,atol=.002))

    def test_teacher_per_image_and_router_gradients(self):
        torch.manual_seed(42)
        source=torch.randn(5,2,8,16)
        target=torch.randn(5,8,16)
        teacher=mllm_teacher_distribution(source,target)
        self.assertEqual(tuple(teacher.shape),(5,2))
        self.assertTrue(torch.allclose(teacher.sum(-1),torch.ones(5)))
        self.assertFalse(teacher.requires_grad)
        model=MLLMHiddenRouter(12,16,hidden=32)
        x=torch.randn(5,12)
        context=torch.randn(5,4)
        prediction,weights=model(x,context,target,return_weights=True)
        self.assertEqual(tuple(prediction.shape),(5,))
        loss=prediction.square().mean()+router_distillation(weights,teacher)
        loss.backward()
        self.assertGreater(float(model.attn.in_proj_weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.router[-1].weight.grad.abs().sum()),0.)

    def test_mask_and_target_token_changes_output(self):
        torch.manual_seed(42)
        model=MLLMHiddenRouter(12,16,hidden=32)
        x=torch.randn(5,12)
        context=torch.randn(5,4)
        target=torch.randn(5,8,16)
        mask=torch.tensor([[True,False]]*5)
        _,weights=model(x,context,target,mask,return_weights=True)
        self.assertTrue(torch.equal(weights[:,1],torch.zeros(5)))
        before=model(x,context,target)
        altered=model(x,context,target.flip(-1))
        self.assertGreater(float((before-altered).abs().max()),1e-8)

    def test_manifest_is_strict_source_only(self):
        rows=[]
        for city in ("A","B","C","D","E","F","G","H"):
            for task in TASKS:
                for _ in range(10):
                    rows.append({"city":city,"indicator_column":task})
        frame=pd.DataFrame(rows)
        m=selection(frame,42,64,12,.25)
        held=set(m["held_cities"])
        self.assertTrue(set(frame.iloc[m["train"]].city).isdisjoint(held))
        self.assertEqual(set(frame.iloc[m["train"]].indicator_column),set(TRAIN_TASKS))
        self.assertTrue(set(frame.iloc[m["UCUI"]].city).issubset(held))
        self.assertTrue(set(frame.iloc[m["SCUI"]].city).isdisjoint(held))
        work=jobs(frame,m)
        self.assertEqual(len(work),3*len(m["train"])+len(m["SCUI"])+len(m["UCUI"]))


if __name__=="__main__":
    unittest.main()
