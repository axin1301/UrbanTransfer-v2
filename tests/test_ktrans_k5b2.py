import unittest
import torch
from urban_transfer.ktrans_normalized import NormalizedExpertBank
from urban_transfer.ktrans_k5b import RegionTaskComposer

class StableAdapterTests(unittest.TestCase):
    def test_preservation_penalty_and_gradients(self):
        torch.manual_seed(13)
        vectors={name:torch.randn(12) for name in ('a','b','c')}
        bank=NormalizedExpertBank(8,vectors,['a','b'],{'a':(1.,1.),'b':(2.,2.)},hidden_dim=16)
        model=RegionTaskComposer(bank,vectors,rank=4,adapter_cap=.5)
        batch={'image':torch.randn(8,8),'context':torch.randn(8,4),
               'task_name':['a','b']*4}
        _,initial=model(batch,exclude_own=True,return_aux=True)
        self.assertAlmostEqual(float(initial['preserve_loss']),0.,places=6)
        with torch.no_grad():
            model.adapter_up.weight.fill_(.4)
        result,aux=model(batch,exclude_own=True,return_aux=True)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreaterEqual(float(aux['preserve_loss']),0.)
        loss=result.square().mean()+aux['preserve_loss']
        loss.backward()
        self.assertIsNotNone(model.adapter_up.weight.grad)
        self.assertFalse(any(p.grad is not None for p in bank.parameters()))

if __name__=='__main__':unittest.main()
