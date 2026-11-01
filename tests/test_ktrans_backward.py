import unittest
import torch
from urban_transfer.ktrans_normalized import NormalizedExpertBank

class RoutingBackwardTest(unittest.TestCase):
    def test_backward_shared_and_separate_scale_routes(self):
        vectors={'a':torch.tensor([1.,0.]),'b':torch.tensor([0.,1.]),
                 'held':torch.tensor([.8,.2])}
        for mode in ('shared','semantic','uniform'):
            with self.subTest(scale_mode=mode):
                model=NormalizedExpertBank(3,vectors,['a','b'],{'a':(2.,1.),'b':(7.,2.)},hidden_dim=8)
                model.set_scale_routing_mode(mode)
                batch={'image':torch.randn(256,3),'context':torch.randn(256,4),
                       'task_name':['a']*128+['b']*128}
                loss=(model(batch)-torch.randn(256)).square().mean()
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.shared.parameters() if p.requires_grad))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                                    for p in model.experts[0].parameters()))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                                    for p in model.experts[1].parameters()))

if __name__=='__main__':unittest.main()
