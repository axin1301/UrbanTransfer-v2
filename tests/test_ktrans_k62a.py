import unittest
from types import SimpleNamespace
import pandas as pd
from scripts.audit_ktrans_k62a import validate_fold,row_keys

class NestedReadinessTests(unittest.TestCase):
    def setUp(self):
        self.pseudo='education_accessibility'
        self.outer='digital_total_cell_intensity'
        self.sources=['education_accessibility','food_accessibility','nighttime_light_intensity']
        records=[
          ('A','a1','food_accessibility'),
          ('A','a2','nighttime_light_intensity'),
          ('A','a3','education_accessibility'),
          ('B','b1','food_accessibility'),
          ('B','b2','nighttime_light_intensity'),
          ('B','b3','education_accessibility'),
        ]
        self.df=pd.DataFrame(records,columns=['city','image_name','indicator_column'])
        self.outer_meta={'heldout_task':self.outer,
                         'source_tasks':self.sources,
                         'heldout_cities':['C']}
        self.inner_meta={'heldout_task':self.pseudo,'heldout_cities':['B']}
        self.inner_model=SimpleNamespace(source_tasks=tuple(self.sources[1:]),
                                         shared=[None,None,None],
                                         experts=[None])
        self.splits={'train':[0,1],'meta_SCUI':[2],
                     'meta_UCSI':[3,4],'meta_UCUI':[5]}

    def test_exact_isolation(self):
        self.inner_model.shared[-2]=SimpleNamespace(out_features=16)
        self.inner_model.experts[0]=[SimpleNamespace(in_features=16)]
        row=validate_fold(self.outer_meta,self.df,self.inner_meta,self.df,
             self.splits,self.inner_model,self.pseudo,'test-last.pt')
        self.assertTrue(row['pseudo_target_expert_excluded'])
        self.assertEqual(row['n_inner_ucui_rows'],1)

    def test_reject_source_leak(self):
        self.inner_model.source_tasks=tuple(self.sources)
        with self.assertRaises(AssertionError):
            validate_fold(self.outer_meta,self.df,self.inner_meta,self.df,
                 self.splits,self.inner_model,self.pseudo,'test-last.pt')

    def test_reject_meta_city_leak(self):
        self.outer_meta['heldout_cities']=['B']
        with self.assertRaises(AssertionError):
            validate_fold(self.outer_meta,self.df,self.inner_meta,self.df,
                 self.splits,self.inner_model,self.pseudo,'test-last.pt')

    def test_reject_row_mismatch(self):
        bad=self.df.copy()
        bad.loc[0,'image_name']='different'
        with self.assertRaises(AssertionError):
            validate_fold(self.outer_meta,self.df,self.inner_meta,bad,
                 self.splits,self.inner_model,self.pseudo,'test-last.pt')

    def test_row_key_identity(self):
        self.assertEqual(len(row_keys(self.df)),len(self.df))

if __name__=='__main__':unittest.main()
