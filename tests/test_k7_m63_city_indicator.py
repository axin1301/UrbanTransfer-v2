"""M6.3 source-only CPU unit tests: frozen visual anchor + bounded transfer."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.run_k7_m62_full import source_train_targets
from scripts.run_k7_m63_full import train_m63,evaluate_m63
from urban_transfer.k7_m1_knowledge import TRAIN_TASKS,HELDOUT_TASK
from urban_transfer.k7_m63_city_indicator_residual import CityIndicatorResidual


class M63CityIndicatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=CityIndicatorResidual(
            visual_dim=12,mllm_dim=24,hidden=32,heads=4,
            group_counts=(6,4,5),max_residual=.5).eval()
        self.x=torch.randn(8,12)
        self.geo=torch.randn(8,4)
        self.names=[TRAIN_TASKS[i%2] for i in range(8)]
        self.y=torch.randn(8)
        class TinyCache:
            def __getitem__(self,key):
                i,task=key
                rng=torch.Generator().manual_seed(
                    91+int(i)*5+
                    (*TRAIN_TASKS,HELDOUT_TASK).index(task))
                return torch.randn(15,24,generator=rng)
            def batch(self,ids,tasks,device):
                return torch.stack([self[int(i),t]
                                    for i,t in zip(ids,tasks)]).to(device)
        self.tokens=TinyCache()
        self.h=self.tokens.batch(
            np.arange(8),self.names,"cpu")
        self.stats={t:(0.,1.) for t in TRAIN_TASKS}

    def test_initial_prediction_exactly_matches_anchor(self):
        with torch.inference_mode():
            pred,route,task,fused,base,delta=self.model(
                self.x,self.geo,self.h,return_aux=True)
        self.assertTrue(torch.equal(pred,base))
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))
        self.assertTrue(torch.allclose(
            base,self.model.base(self.x,self.geo),atol=0,rtol=0))
        self.assertTrue(torch.allclose(
            route.sum(-1),torch.ones(8),atol=1e-6))
        self.assertTrue(all(not p.requires_grad
                            for p in self.model.base.parameters()))

    def test_bounded_residual_and_city_indicator_effect(self):
        with torch.no_grad():
            self.model.residual_scale.weight.fill_(4.)
            p,w,_,_,base,d=self.model(
                self.x,self.geo,self.h,return_aux=True)
            self.assertTrue(bool((d.abs()<=.5000001).all()))
            pop=self.tokens.batch(np.arange(8),
                                  [TRAIN_TASKS[0]]*8,"cpu")
            edu=self.tokens.batch(np.arange(8),
                                  [HELDOUT_TASK]*8,"cpu")
            qpop=self.model.condition(self.x,self.geo,pop)[2]
            qedu=self.model.condition(self.x,self.geo,edu)[2]
            self.assertGreater(float((qpop-qedu).abs().mean()),1e-5)
            flipped=self.model(self.x.flip(0),self.geo,edu)
            self.assertFalse(torch.allclose(p,flipped))

    def test_source_expert_and_auxiliary_terms(self):
        pop=self.tokens.batch(np.arange(8),[TRAIN_TASKS[0]]*8,"cpu")
        food=self.tokens.batch(np.arange(8),[TRAIN_TASKS[1]]*8,"cpu")
        edu=self.tokens.batch(np.arange(8),[HELDOUT_TASK]*8,"cpu")
        source=self.model.source_expert_predictions(
            self.x,self.geo,self.h)
        classification=self.model.source_task_separation(
            self.x,self.geo,pop,food)
        align=self.model.target_image_alignment(
            self.x,self.geo,edu)
        self.assertEqual(tuple(source.shape),(8,2))
        self.assertTrue(bool(torch.isfinite(classification)))
        self.assertTrue(bool(torch.isfinite(align)))

    def test_source_fit_never_modifies_anchor(self):
        state={k:v.detach().clone()
               for k,v in self.model.base.state_dict().items()}
        class Args:
            seed=42
            epochs=1
            batch_size=4
            lr=.001
            self_loss=.3
            task_loss=.2
            target_align_loss=.05
            residual_penalty=.02
        train_m63(
            self.model,self.x,self.geo,self.y,self.names,
            np.arange(8),self.stats,self.tokens,
            Args(),torch.device("cpu"))
        for name,tensor in state.items():
            self.assertTrue(torch.equal(
                tensor,self.model.base.state_dict()[name]))
        output=evaluate_m63(
            self.model,"UCSI",np.arange(8),self.x,self.geo,self.y,
            self.names,self.stats,(0.,1.),self.tokens,
            4,torch.device("cpu"))
        self.assertEqual(output["n_rows"],8)
        self.assertTrue(np.isfinite(output["mse_log1p"]))
        self.assertLessEqual(
            output["max_abs_transfer_normalized"],.500001)

    def test_task_and_context_shape_guard(self):
        with self.assertRaises(ValueError):
            self.model(self.x,self.geo,torch.randn(8,14,24))
        with self.assertRaises(ValueError):
            self.model(self.x,torch.randn(8,3),self.h)

    def test_pretrained_base_round_trip(self):
        reference=CityIndicatorResidual(
            12,24,hidden=32,heads=4,
            group_counts=(6,4,5))
        reference.load_visual_baseline(
            self.model.base.state_dict())
        with torch.inference_mode():
            self.assertTrue(torch.allclose(
                reference.base(self.x,self.geo),
                self.model.base(self.x,self.geo)))
        self.assertTrue(all(not p.requires_grad
                            for p in reference.base.parameters()))


if __name__=="__main__":
    unittest.main()
