import unittest
import torch
from urban_transfer.ktrans_experts import SemanticExpertBank

class RoutingAblationTests(unittest.TestCase):
    def setUp(self):
        self.model=SemanticExpertBank(8, {
            's1':torch.tensor([1.,0.,0.]),'s2':torch.tensor([0.,1.,0.]),
            's3':torch.tensor([0.,0.,1.]),'held':torch.tensor([1.,.2,0.])
        },['s1','s2','s3'],hidden_dim=16)
        self.model.eval()
        self.batch={'image':torch.randn(4,8),'context':torch.randn(4,4),
                    'task_name':['held','held','s1','s2']}
    def test_identical_parameters_for_all_modes(self):
        before={k:v.detach().clone() for k,v in self.model.state_dict().items()}
        outputs=[]
        for mode in ('semantic','uniform','random'):
            self.model.set_routing_mode(mode,seed=10)
            outputs.append(self.model(self.batch,None).detach().clone())
        for k,v in before.items():
            self.assertTrue(torch.equal(v,self.model.state_dict()[k]))
        self.assertTrue(torch.allclose(outputs[0][2:],outputs[1][2:]))
        self.assertTrue(torch.allclose(outputs[1][2:],outputs[2][2:]))
    def test_random_is_reproducible(self):
        self.model.set_routing_mode('random',seed=123)
        x=self.model(self.batch,None).detach().clone()
        self.model.set_routing_mode('random',seed=123)
        self.assertTrue(torch.equal(x,self.model(self.batch,None)))
    def test_uniform_weights(self):
        self.model.set_routing_mode('uniform')
        out=self.model(self.batch,None)
        with torch.no_grad():
            h=self.model.shared(torch.cat([self.batch['image'],self.batch['context']],dim=-1))
            expected=torch.stack([e(h).squeeze(-1) for e in self.model.experts],dim=1).mean(1)
        self.assertTrue(torch.allclose(out[:2],expected[:2],atol=1e-6))

if __name__=='__main__':unittest.main()
