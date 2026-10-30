import unittest
import pandas as pd
from urban_transfer.data_m1 import stratified_sample

class StratifiedTrainingLimitTests(unittest.TestCase):
    def setUp(self):
        self.df=pd.DataFrame([{'city':city,'indicator_column':task,'row':i}
            for task in ['population_count','food_accessibility','education_accessibility']
            for city in ['A','B','C','D']
            for i in range(100)])
    def test_full_coverage_reproducible_without_duplicates(self):
        x=stratified_sample(self.df,60,seed=42)
        y=stratified_sample(self.df,60,seed=42)
        self.assertEqual(len(x),60)
        self.assertEqual(x.row.tolist(),y.row.tolist())
        self.assertEqual(x.indicator_column.nunique(),3)
        self.assertEqual(x.city.nunique(),4)
        self.assertEqual(x.groupby(['city','indicator_column']).size().min(),5)
        self.assertEqual(x.groupby(['city','indicator_column']).size().max(),5)
        self.assertFalse(x.index.duplicated().any())
    def test_too_small_raises(self):
        with self.assertRaisesRegex(ValueError,'smaller than'):
            stratified_sample(self.df,5)
    def test_full_data_unchanged(self):
        self.assertEqual(len(stratified_sample(self.df,None)),len(self.df))

if __name__=='__main__':unittest.main()
