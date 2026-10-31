import unittest
import pandas as pd
from urban_transfer.ktrans import meta_indices

class MetaProtocolTests(unittest.TestCase):
    def setUp(self):
        self.df=pd.DataFrame([
            {'city':city,'indicator_column':task,'image_name':f'{city}-{i}'}
            for city in ['A','B','C','D','E']
            for task in ['population_count','digital_total_cell_intensity','food_accessibility']
            for i in range(3)])
    def test_city_task_disjoint_and_deterministic(self):
        first,cities=meta_indices(self.df,'digital_total_cell_intensity',.2,42)
        second,cities2=meta_indices(self.df,'digital_total_cell_intensity',.2,42)
        self.assertEqual(first,second)
        self.assertEqual(cities,cities2)
        self.assertEqual(sum(map(len,first.values())),len(self.df))
        self.assertEqual(len(set(sum(first.values(),[]))),len(self.df))
        for idx in first['train']:
            row=self.df.iloc[idx]
            self.assertNotEqual(row.indicator_column,'digital_total_cell_intensity')
            self.assertNotIn(row.city,cities)
        for idx in first['meta_UCUI']:
            row=self.df.iloc[idx]
            self.assertEqual(row.indicator_column,'digital_total_cell_intensity')
            self.assertIn(row.city,cities)
    def test_bad_fraction(self):
        with self.assertRaises(ValueError):
            meta_indices(self.df,'digital_total_cell_intensity',0,42)

if __name__=='__main__':unittest.main()
