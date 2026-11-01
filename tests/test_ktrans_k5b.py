import unittest
import torch
from urban_transfer.ktrans_normalized import NormalizedExpertBank
from urban_transfer.ktrans_k5b import RegionTaskComposer

class K5BTests(unittest.TestCase):
    def test_initialization_and_gradient(self):
        torch.manual_seed(0)
        tasks=['a','b','c']
        vec={t:torch.nn.functional.normalize(torch.randn(16),dim=0) for t in tasks}
        stats={'a':(1.,1.),'b':(2.,2.)}
        bank=NormalizedExpertBank(8,vec,['a','b'],stats,hidden_dim=16)
        model=RegionTaskComposer(bank,vec,rank=4)
        batch={'image':torch.randn(5,8),
               'context':torch.randn(5,4),
               'task_name':['a','a','b','b','a']}
        # Evaluation starts exactly at frozen semantic K4 prediction.
        with torch.no_grad():
            unseen={**batch,'task_name':['c']*5}
            self.assertTrue(torch.allclose(model(unseen),bank(unseen),atol=1e-6))
        out,aux=model(batch,exclude_own=True,return_aux=True)
        self.assertTrue((aux['weights'][0,0]<1e-6).item())
        self.assertTrue((aux['weights'][2,1]<1e-6).item())
        loss=out.square().mean()
        loss.backward()
        self.assertIsNotNone(model.route[-1].weight.grad)
        self.assertFalse(any(p.grad is not None for p in bank.parameters()))

    def test_unseen_task_uses_all_experts(self):
        vec={t:torch.randn(12) for t in ('a','b','c')}
        bank=NormalizedExpertBank(8,vec,['a','b'],{'a':(1.,1.),'b':(3.,2.)},hidden_dim=16)
        model=RegionTaskComposer(bank,vec)
        batch={'image':torch.randn(2,8),'context':torch.zeros(2,4),
               'task_name':['c','c']}
        pred,aux=model(batch,return_aux=True)
        self.assertTrue(torch.isfinite(pred).all())
        self.assertTrue((aux['weights']>0).all())
        self.assertTrue(torch.allclose(aux['weights'].sum(-1),torch.ones(2)))

if __name__=='__main__':unittest.main()
