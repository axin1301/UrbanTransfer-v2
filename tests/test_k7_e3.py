"""CPU-only K7 E3 unit tests: relation binding, rewards and split leakage."""
import unittest
import numpy as np
import pandas as pd
import torch

from urban_transfer.k7_utility import (
    fit_ridge, inner_city_split, pair_features, predicted_weights)


class K7E3Tests(unittest.TestCase):
    def test_pairwise_relation_shuffle_actually_changes_features(self):
        torch.manual_seed(4)
        tasks = list("abcdef")
        semantic = {name: torch.randn(32) for name in tasks}
        relation = {(a,b): torch.randn(32)
                    for a in tasks for b in tasks if a != b}
        sources = list("acde")
        normal = pair_features("a","b",semantic,relation,sources,False)
        shuffled = pair_features("a","b",semantic,relation,sources,True)
        self.assertFalse(np.allclose(normal,shuffled))
        self.assertEqual(len(normal),6)

    def test_utility_weights_include_no_transfer_and_normalize(self):
        torch.manual_seed(3)
        tasks=list("abcdef")
        sem={k:torch.randn(32) for k in tasks}
        rel={(a,b):torch.randn(32) for a in tasks for b in tasks if a!=b}
        rewards={(t,s): float((i+j)%4)/10-.15
                 for i,t in enumerate("abcde")
                 for j,s in enumerate("abcde") if t!=s}
        w,p=predicted_weights("f", list("abcde"), rewards, sem, rel)
        ws,ps=predicted_weights("f", list("abcde"), rewards, sem, rel, shuffle=True)
        self.assertEqual(len(w),6)
        self.assertAlmostEqual(float(w.sum()),1.)
        self.assertTrue(np.isfinite(w).all())
        self.assertFalse(np.allclose(p,ps))

    def test_inner_validation_never_uses_outer_held_cities(self):
        rows=[{"city":city,"indicator_column":task}
              for city in ["A","B","C","D","E","F"]
              for task in ["source_1","source_2"]]
        frame=pd.DataFrame(rows)
        outer=np.flatnonzero(frame.city.ne("F").to_numpy())
        train,val=inner_city_split(frame,outer,42,.2)
        self.assertFalse(set(train)&set(val))
        self.assertTrue(set(train).issubset(set(outer)))
        self.assertTrue(set(val).issubset(set(outer)))
        self.assertTrue(set(frame.iloc[train].city).isdisjoint(frame.iloc[val].city))
        self.assertNotIn("F",set(frame.iloc[val].city))

    def test_ridge_handles_small_source_pair_supervision(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(20,6))
        x[:,0]=1.0
        targets=rng.normal(size=20)
        w=fit_ridge(x,targets)
        self.assertEqual(tuple(w.shape),(6,))
        self.assertTrue(np.isfinite(w).all())


if __name__ == "__main__":
    unittest.main()
