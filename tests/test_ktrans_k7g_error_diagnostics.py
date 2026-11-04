import unittest
import numpy as np
import torch
from urban_transfer.ktrans_k7g import error_decomposition,direction_summary
from scripts.run_ktrans_k7g_error_diagnostics import context

class K7GErrorDiagnosticsTests(unittest.TestCase):
    def test_exact_bias_and_centering_identity(self):
        y=np.asarray([0.,1.,2.,3.])
        p=y+2.
        result=error_decomposition(y,p,y)
        self.assertAlmostEqual(result['mse'],4.)
        self.assertAlmostEqual(result['bias_squared'],4.)
        self.assertAlmostEqual(result['centered_residual_mse'],0.)
        self.assertAlmostEqual(result['delta_mse_vs_k5a'],4.)
        self.assertAlmostEqual(result['paired_linear_error_alignment'],0.)
        self.assertAlmostEqual(result['paired_quadratic_perturbation'],4.)

    def test_affine_oracle_in_sample_not_reportable(self):
        y=np.asarray([0.,1.,2.,3.])
        p=3*y+5
        out=error_decomposition(y,p)
        self.assertLess(out['in_sample_oracle_mse_NOT_GENERALIZATION'],1e-10)
        self.assertAlmostEqual(out['in_sample_oracle_slope'],1/3)
        self.assertAlmostEqual(out['pearson'],1.)

    def test_paired_improvement_and_alignment(self):
        y=np.asarray([0.,1.,2.,3.])
        b=y+1.
        p=y+.5
        z=error_decomposition(y,p,b)
        self.assertAlmostEqual(z['delta_mse_vs_k5a'],-.75)
        self.assertAlmostEqual(z['paired_linear_error_alignment'],-1.)
        self.assertAlmostEqual(z['paired_quadratic_perturbation'],.25)
        self.assertAlmostEqual(z['delta_bias_squared'],-.75)
        self.assertAlmostEqual(z['delta_centered_mse'],0.)

    def test_constant_prediction_oracle_safe(self):
        y=np.asarray([0.,1.,2.])
        z=error_decomposition(y,np.ones(3))
        self.assertTrue(np.isnan(z['in_sample_oracle_slope']))
        self.assertAlmostEqual(z['in_sample_oracle_mse_NOT_GENERALIZATION'],np.var(y))

    def test_source_direction_is_explicitly_in_sample(self):
        source={'variant':'image_task','gate_mode':'trained','gate':.12,
                'delta_mse_vs_k5a':-.1}
        eval={'variant':'image_task','gate_mode':'trained','gate':.12,
              'delta_mse_vs_k5a':.2}
        result=direction_summary(source,eval)
        self.assertFalse(result['direction_matches'])
        self.assertEqual(result['source_train_direction'],'better')
        self.assertEqual(result['target_dev_direction'],'worse')

    def test_k7d_source_own_expert_mask(self):
        class Bank:
            source_tasks=('a','b','c')
            task_to_source={'a':0,'b':1,'c':2}
            task_to_all={'a':0,'b':1,'c':2,'held':3}
            routing=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[.2,.3,.5]])
            source_means=torch.tensor([1.,2.,3.])
            source_stds=torch.tensor([1.,1.,1.])
        source=torch.tensor([[0.,.5,.5],[.5,0.,.5],[.5,.5,0.]])
        rows=[{'task':'a'},{'task':'held'}]
        prior,mask,mean,std=context(rows,Bank(),source,torch.tensor([.2,.3,.5]),torch.device('cpu'))
        self.assertTrue(mask[0,0])
        self.assertFalse(mask[1].any())
        self.assertEqual(float(prior[0,0]),0.)
        self.assertAlmostEqual(float(mean[0]),1.)
        self.assertAlmostEqual(float(mean[1]),2.3,places=5)

if __name__=='__main__':unittest.main()
