import unittest
import torch
from urban_transfer.ktrans_normalized import NormalizedExpertBank

class TestIndependentRouting(unittest.TestCase):
    def setUp(self):
        vectors={'a':torch.tensor([1.,0.]),'b':torch.tensor([0.,1.]),
                 'unseen':torch.tensor([.9,.1])}
        self.model=NormalizedExpertBank(3,vectors,['a','b'],
                  {'a':(2.,1.),'b':(10.,2.)},hidden_dim=8)
        for expert in self.model.experts:
            for p in expert.parameters():p.data.zero_()
    def test_scale_route_only_changes_unseen_mean(self):
        batch={'image':torch.zeros(2,3),'context':torch.zeros(2,4),
               'task_name':['unseen','a']}
        self.model.set_routing_mode('semantic')
        self.model.set_scale_routing_mode('semantic')
        sem=self.model(batch)
        self.model.set_scale_routing_mode('uniform')
        uniform=self.model(batch)
        self.assertAlmostEqual(float(uniform[0]),6.,places=5)
        self.assertAlmostEqual(float(sem[1]),2.,places=5)
        self.assertAlmostEqual(float(uniform[1]),2.,places=5)
        self.assertGreater(abs(float(sem[0])-float(uniform[0])),.1)
    def test_reject_unsupported_scale_mode(self):
        with self.assertRaises(ValueError):
            self.model.set_scale_routing_mode('oracle')

if __name__=='__main__':unittest.main()
