import unittest
import numpy as np
import torch
from scripts.audit_ktrans_inputs import summarize_inputs,cosine_matrix

class InputAuditTests(unittest.TestCase):
    def test_input_feature_statistics(self):
        context=np.array([[.1,.2,.3,1.],[.2,.4,.3,1.]],dtype=float)
        image=np.array([[1.,0.,0.],[0.,1.,0.]],dtype=float)
        row=summarize_inputs(context,image,[1.,2.])
        self.assertEqual(row['image_dim'],3)
        self.assertEqual(row['unique_visual_rounded_6'],2)
        self.assertAlmostEqual(row['intercept_std'],0.)
        self.assertAlmostEqual(row['source_label_mean'],1.5)

    def test_invalid_embeddings_rejected(self):
        vectors={'a':torch.tensor([0.,0.]),'b':torch.tensor([1.,0.])}
        with self.assertRaises(ValueError):cosine_matrix(vectors,['a','b'])

    def test_similarity_matrix(self):
        v={'a':torch.tensor([1.,0.]),'b':torch.tensor([0.,1.])}
        m=cosine_matrix(v,['a','b'])
        np.testing.assert_allclose(m,np.eye(2),atol=1e-7)

    def test_nonfinite_input_rejected(self):
        with self.assertRaises(ValueError):
            summarize_inputs([[float('nan'),0,0,1]],[[1.,2.]])

if __name__=='__main__':unittest.main()
