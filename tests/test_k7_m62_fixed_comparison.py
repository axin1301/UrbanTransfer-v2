"""CPU tests for fixed M6.2 matched UCSI + baseline comparison."""
import unittest
import numpy as np
import pandas as pd
import torch

from scripts.cache_k7_m62_ucsi import select_ucsi
from scripts.run_k7_m62_fixed_comparison import (
    TwoExpertControl,assert_protocol,eval_split)
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK


class FixedComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_ucsi_selection_held_city_source_only_deterministic(self):
        rows=[]
        for city in ("A","B","C","D"):
            for task in (*TRAIN_TASKS,HELDOUT_TASK):
                for row in range(8):
                    rows.append({"city":city,"indicator_column":task})
        frame=pd.DataFrame(rows)
        manifest={"held_cities":["C","D"],
                  "train":[0,8],
                  "SCUI":[16],"UCUI":[64+16]}
        first=select_ucsi(frame,manifest,limit=12,seed=42)
        second=select_ucsi(frame,manifest,limit=12,seed=42)
        self.assertEqual(first,second)
        self.assertEqual(len(first),12)
        subset=frame.iloc[first]
        self.assertTrue(subset.city.isin(("C","D")).all())
        self.assertEqual(set(subset.indicator_column),set(TRAIN_TASKS))

    def test_source_visual_only_never_consumes_text(self):
        model=TwoExpertControl(12,hidden=32).eval()
        x=torch.randn(7,12)
        ctx=torch.randn(7,4)
        with torch.no_grad():
            pred,w=model(x,ctx,return_aux=True)
            specialists=model.source_expert_predictions(x,ctx)
        self.assertEqual(tuple(pred.shape),(7,))
        self.assertEqual(tuple(w.shape),(7,2))
        self.assertEqual(tuple(specialists.shape),(7,2))
        self.assertTrue(torch.allclose(w.sum(-1),torch.ones(7)))
        with self.assertRaises(ValueError):
            model(x,ctx,torch.randn(7,8))

    def test_text_router_responds_to_target_semantics(self):
        model=TwoExpertControl(12,text_dim=16,hidden=32).eval()
        x=torch.randn(7,12)
        ctx=torch.randn(7,4)
        with torch.no_grad():
            a,wa=model(x,ctx,torch.randn(7,16),return_aux=True)
            b,wb=model(x,ctx,torch.randn(7,16),return_aux=True)
        self.assertGreater(float((a-b).abs().mean()),1e-7)
        self.assertGreater(float((wa-wb).abs().mean()),1e-7)

    def test_eval_split_task_specific_source_scaling(self):
        model=TwoExpertControl(12,hidden=32).eval()
        x=torch.randn(5,12)
        ctx=torch.randn(5,4)
        y=torch.tensor([0.5,1.,1.5,2.,2.5])
        names=[TRAIN_TASKS[0],TRAIN_TASKS[1],HELDOUT_TASK,
               TRAIN_TASKS[0],TRAIN_TASKS[1]]
        stats={TRAIN_TASKS[0]:(1.,2.),
               TRAIN_TASKS[1]:(2.,3.)}
        output=eval_split("visual_only",model,"UCSI",
                          np.asarray([0,1,3,4],dtype=np.int64),
                          x,ctx,y,names,stats,(2.,1.),
                          {},torch.device("cpu"))
        self.assertEqual(output["n_rows"],4)
        self.assertTrue(np.isfinite(output["mse_log1p"]))
        self.assertTrue(np.isfinite(output["relative_mse"]))

    def test_protocol_rejects_ucsi_target_and_overlap(self):
        frame=pd.DataFrame({
            "city":["A","A","B","B","B","A"],
            "indicator_column":[TRAIN_TASKS[0],HELDOUT_TASK,
                                TRAIN_TASKS[0],TRAIN_TASKS[1],
                                HELDOUT_TASK,TRAIN_TASKS[1]]})
        manifest={"train":[0,5],"SCUI":[1],"UCUI":[4],
                  "held_cities":["B"]}
        new={"UCSI":[2,3],"source_tasks":list(TRAIN_TASKS),
             "held_cities":["B"]}
        # Fixed pilot requires 64 UCSI; do not relax protocol.
        with self.assertRaises(ValueError):
            assert_protocol(frame,manifest,new)
        with self.assertRaises(ValueError):
            assert_protocol(frame,manifest,{**new,"UCSI":[1]*64})


if __name__=="__main__":
    unittest.main()
