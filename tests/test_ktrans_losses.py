import unittest
import torch
from urban_transfer.ktrans_losses import (
    grouped_correlation_loss,city_pairwise_rank_loss,transfer_loss)

class TransferLossTests(unittest.TestCase):
    def setUp(self):
        self.y=torch.tensor([0.,1.,2.,3.,4.,5.,6.,7., 0.,1.,2.,3.,4.,5.,6.,7.])
        self.tasks=['education_accessibility']*8+['food_accessibility']*8
        self.cities=['City A']*8+['City B']*8

    def test_grouped_corr_and_gradients(self):
        pred=(self.y+torch.linspace(0.,.5,len(self.y))).clone().requires_grad_()
        loss=grouped_correlation_loss(pred,self.y,self.tasks)
        self.assertGreaterEqual(float(loss),0.)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_perfect_correlation_never_negative(self):
        # Regression: float32 reductions may otherwise give Pearson > 1.
        y=torch.arange(0.,32.)
        tasks=['education_accessibility']*len(y)
        for scale in (1.,1.1,2.3):
            result=grouped_correlation_loss(y*scale,y,tasks)
            self.assertGreaterEqual(float(result),0.)
            self.assertLess(float(result),1e-5)

    def test_corr_does_not_compare_different_tasks(self):
        pred=self.y.clone()
        self.assertAlmostEqual(float(grouped_correlation_loss(
            pred,self.y,self.tasks)),0.,places=5)

    def test_rank_prefers_correct_order(self):
        good=self.y.clone().requires_grad_()
        bad=(-self.y).clone().requires_grad_()
        lgood=city_pairwise_rank_loss(good,self.y,self.tasks,self.cities)
        lbad=city_pairwise_rank_loss(bad,self.y,self.tasks,self.cities)
        self.assertLess(float(lgood),float(lbad))
        lgood.backward()
        self.assertIsNotNone(good.grad)

    def test_rank_no_cross_city_pairs(self):
        pred=torch.tensor([1.,-1.],requires_grad=True)
        loss=city_pairwise_rank_loss(pred,torch.tensor([0.,5.]),
                   ['education_accessibility']*2,['A','B'])
        self.assertEqual(float(loss),0.)

    def test_modes_and_default_mse(self):
        pred=(self.y+.25).clone().requires_grad_()
        target=self.y
        default,parts=transfer_loss(pred,target,self.tasks,self.cities)
        self.assertAlmostEqual(float(default),float((pred-target).square().mean()))
        for mode in ('mse','corr','rank','combined'):
            loss,_=transfer_loss(pred,target,self.tasks,self.cities,mode)
            self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):
            transfer_loss(pred,target,self.tasks,self.cities,'unknown')

if __name__=='__main__':unittest.main()
