import unittest
import torch
from scripts.diagnose_ktrans_loss_gradients import (
    group_coverage,gradient,cosine,analyze_batch)

class CoverageTests(unittest.TestCase):
    def test_coverage_respects_task_and_city(self):
        y=torch.arange(16,dtype=torch.float32)
        tasks=['taskA']*8+['taskB']*8
        cities=['A']*8+['B']*8
        stats=group_coverage(y,tasks,cities)
        self.assertEqual(stats['corr_groups'],2)
        self.assertEqual(stats['corr_samples'],16)
        self.assertEqual(stats['rank_groups'],2)
        self.assertEqual(stats['rank_pairs'],8)

    def test_no_pairs_across_city(self):
        x=torch.tensor([0.,5.])
        stats=group_coverage(x,['a','a'],['X','Y'])
        self.assertEqual(stats['rank_pairs'],0)
        self.assertEqual(stats['corr_groups'],0)

    def test_gradient_cosine(self):
        w=torch.nn.Parameter(torch.tensor([1.,2.]))
        x=gradient((w*w).sum(),[w])
        y=gradient(-(w*w).sum(),[w])
        self.assertAlmostEqual(cosine(x,y),-1.)
        self.assertAlmostEqual(cosine(x,torch.zeros_like(x)),float('nan')) if False else None

    def test_zero_gradient_loss(self):
        w=torch.nn.Parameter(torch.tensor([1.,2.]))
        result=gradient((w*0.).sum(),[w])
        self.assertEqual(float(result.abs().sum()),0.)

if __name__=='__main__':unittest.main()
