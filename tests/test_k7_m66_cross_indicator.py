"""M6.6 CPU synthetic tests for explicit neighbor × indicator interaction."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from urban_transfer.k7_m66_indicator_graph import (
    physical_neighbor_graph,cached_physical_neighbor_graph)
from urban_transfer.k7_m66_cross_indicator import NeighborIndicatorInteraction
from scripts.run_k7_m66_cross_indicator_dev import source_prompt_prototypes


class M66Tests(unittest.TestCase):
    def setUp(self):
        rows=[]
        for city in ("A","B"):
            for tile in range(4):
                for name in ("population_count","food_accessibility","education"):
                    rows.append({
                        "city":city,"image_name":f"{tile}.png",
                        "indicator_column":name,
                        "patch_lon":float(tile)*.01+ord(city)*.1,
                        "patch_lat":55.+ord(city)*.001,
                        "indicator_value_log1p":float(tile)})
        self.frame=pd.DataFrame(rows)
        self.visual=torch.randn(len(rows),12)
        for i in range(0,len(rows),3):
            self.visual[i+1:i+3]=self.visual[i]

    def test_graph_has_no_self_and_consistent_duplicate_tiles(self):
        idx,dist,mask=physical_neighbor_graph(self.frame,k=3)
        self.assertEqual(tuple(idx.shape),(len(self.frame),3))
        for i in range(0,len(self.frame),3):
            self.assertTrue(torch.equal(idx[i],idx[i+1]))
            self.assertTrue(torch.equal(dist[i],dist[i+2]))
            self.assertTrue(bool(mask[i].all()))
            self.assertFalse(bool((idx[i]==i).any()))
            self.assertTrue(bool(
                (self.frame.city.iloc[idx[i].numpy()]==
                 self.frame.city.iloc[i]).all()))
        changed=self.frame.copy()
        changed["indicator_value_log1p"]=1e9
        changed["indicator_column"]="unknown"
        graph=physical_neighbor_graph(changed,k=3)
        self.assertTrue(torch.equal(graph[0],idx))
        with tempfile.TemporaryDirectory() as folder:
            m=Path(folder)/"manifest.json"
            m.write_text('{"seed":42}')
            out=Path(folder)/"graph.pt"
            a=cached_physical_neighbor_graph(self.frame,m,out,k=3)
            b=cached_physical_neighbor_graph(self.frame,m,out,k=3)
            self.assertTrue(torch.equal(a[0],b[0]))

    def test_interaction_modes_same_init_and_backward(self):
        torch.manual_seed(42)
        prototype=torch.randn(3,48)
        x=self.visual[:6]
        context=torch.randn(6,4)
        tokens=torch.randn(6,15,24)
        nb=torch.randn(6,38)
        graph,distance,mask=physical_neighbor_graph(self.frame,k=3)
        neighbors=self.visual[graph[:6]]
        results=[]
        for mode in ("cross_indicator","spatial_only"):
            torch.manual_seed(42)
            model=NeighborIndicatorInteraction(
                12,24,prototype,mode=mode,hidden=32,heads=4,
                group_counts=(6,4,5))
            model.eval()
            pred,w,task,fused,base,delta=model(
                x,context,tokens,nb,neighbors,
                distance[:6],mask[:6],return_aux=True)
            self.assertTrue(torch.equal(pred,base))
            self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))
            self.assertEqual(tuple(w.shape),(6,2))
            own=model.source_expert_predictions(
                x,context,tokens,nb,neighbors,distance[:6],mask[:6])
            loss=(pred.square().mean()+own.square().mean()+
                  model.source_task_separation(
                      x,context,tokens,tokens+1,nb,
                      neighbors,distance[:6],mask[:6])+
                  model.target_image_alignment(
                      x,context,tokens,nb,neighbors,
                      distance[:6],mask[:6]))
            loss.backward()
            self.assertIsNotNone(
                model.neighbor_semantic_project[1].weight.grad)
            self.assertTrue(all(not p.requires_grad
                                for p in model.base.parameters()))
            results.append({
                k:v.clone() for k,v in model.state_dict().items()})
        for key in results[0]:
            self.assertTrue(torch.equal(results[0][key],results[1][key]),
                            msg=key)

    def test_invalid_graph_mask_is_rejected(self):
        model=NeighborIndicatorInteraction(
            12,24,torch.randn(3,48),mode="cross_indicator",
            hidden=32,heads=4,group_counts=(6,4,5))
        with self.assertRaisesRegex(ValueError,"physical neighbor"):
            model(torch.randn(2,12),torch.randn(2,4),
                  torch.randn(2,15,24),torch.randn(2,38),
                  torch.randn(2,3,12),torch.zeros(2,3),
                  torch.zeros(2,3,dtype=torch.bool))


if __name__=="__main__":
    unittest.main()
