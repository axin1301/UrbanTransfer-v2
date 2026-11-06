"""CPU protocol tests for full-data M6.2 manifest and batched models.

Tests do not require GPU, images, SQLite or source CSV content.
"""
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.prepare_k7_m62_full import split_full,build_job_index
from scripts.run_k7_m62_full import (
    validate_full,source_train_targets,do_fit,evaluate)
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK


class FullM62Tests(unittest.TestCase):
    def setUp(self):
        rows=[]
        for city in ("A","B","C","D","E","F","G","H"):
            for tile in range(12):
                for indicator in (*TRAIN_TASKS,HELDOUT_TASK):
                    rows.append({
                        "city":city,"image_name":f"{city}_{tile:02d}.png",
                        "indicator_column":indicator,
                        "indicator_value_log1p":float(tile)})
        self.frame=pd.DataFrame(rows)

    def test_source_train_val_disjoint_at_tile_level(self):
        a=split_full(self.frame,42,.25,.20)
        b=split_full(self.frame,42,.25,.20)
        self.assertEqual(a,b)
        splits=validate_full(self.frame,a)
        train_keys=set(zip(
            self.frame.iloc[splits["train"]].city,
            self.frame.iloc[splits["train"]].image_name))
        val_keys=set(zip(
            self.frame.iloc[splits["source_val"]].city,
            self.frame.iloc[splits["source_val"]].image_name))
        self.assertFalse(train_keys&val_keys)
        held=set(a["held_cities"])
        self.assertFalse(set(self.frame.iloc[splits["train"]].city)&held)
        self.assertTrue(set(self.frame.iloc[splits["UCUI"]].city)<=held)
        tasks=set(self.frame.iloc[splits["UCUI"]].indicator_column)
        self.assertEqual(tasks,{HELDOUT_TASK})

    def test_manifest_cache_jobs_include_train_three_tasks(self):
        a=split_full(self.frame)
        work=build_job_index(self.frame,a)
        jobs=set(work)
        self.assertEqual(
            len(work),
            3*len(a["train"])+len(a["source_val"])+
            len(a["SCUI"])+len(a["UCSI"])+len(a["UCUI"]))
        for i in a["train"]:
            for task in (*TRAIN_TASKS,HELDOUT_TASK):
                self.assertIn((i,task),jobs)

    def test_overlap_is_rejected(self):
        a=split_full(self.frame)
        a["UCSI"]=[a["train"][0]]+a["UCSI"]
        with self.assertRaises(ValueError):
            validate_full(self.frame,a)

    def test_training_targets_calibrated_from_source_only(self):
        ids=np.array([0,1,3,4])
        names=[TRAIN_TASKS[0],TRAIN_TASKS[1],
               HELDOUT_TASK,TRAIN_TASKS[0],TRAIN_TASKS[1]]
        y=torch.tensor([4.,5.,6.,7.,8.])
        z=source_train_targets(ids,names,y,{
            TRAIN_TASKS[0]:(3.,2.),
            TRAIN_TASKS[1]:(5.,3.)})
        self.assertTrue(torch.allclose(
            z,torch.tensor([.5,0.,2.,1.])))

    def test_cpu_models_fit_minibatches_without_target_labels(self):
        # Minimal end-to-end training exercise, no actual image dataset.
        class FakeTokenCache:
            def __getitem__(self,key):
                i,task=key
                g=torch.Generator().manual_seed(
                    100+int(i)*3+(*TRAIN_TASKS,HELDOUT_TASK).index(task))
                return torch.randn(15,24,generator=g)
            def batch(self,indices,tasks,device):
                return torch.stack([self[int(i),t]
                                    for i,t in zip(indices,tasks)]).to(device)
        class Args:
            seed=42
            epochs=1
            batch_size=4
            hidden=32
            heads=4
            lr=.001
            self_loss=.3
            task_loss=.2
            target_align_loss=.05
        names=[TRAIN_TASKS[i%2] for i in range(8)]
        x=torch.randn(8,12)
        ctx=torch.randn(8,4)
        y=torch.randn(8)
        embeddings={t:torch.randn(16) for t in (*TRAIN_TASKS,HELDOUT_TASK)}
        stats={t:(0.,1.) for t in TRAIN_TASKS}
        cache=FakeTokenCache()
        manifest={"group_counts":[6,4,5]}
        for variant in ("visual_only","text_router","m62"):
            model=do_fit(variant,manifest,x,ctx,y,names,
                         np.arange(8),stats,embeddings,
                         cache,Args(),torch.device("cpu"))
            row=evaluate(variant,model,"UCSI",np.arange(8),
                         x,ctx,y,names,stats,(0.,1.),embeddings,
                         cache,4,torch.device("cpu"))
            self.assertEqual(row["n_rows"],8)
            self.assertTrue(np.isfinite(row["mse_log1p"]))


if __name__=="__main__":
    unittest.main()
