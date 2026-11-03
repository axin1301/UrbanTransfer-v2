import unittest
import torch
from urban_transfer.measurement_knowledge import relatedness,MEASUREMENT_ATTRIBUTES
from urban_transfer.ktrans_k5d import MeasurementAwareScaleRouter,moment_estimate

class K5DTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        tasks=('population_count','nighttime_light_intensity',
               'education_accessibility','food_accessibility',
               'digital_modernization_tmi','digital_total_cell_intensity',
               'healthcare_accessibility')
        self.vec={t:torch.randn(24) for t in tasks}
        self.stats={name:(float(i+1),float(i+.5)) for i,name in
           enumerate(tasks[:6],1)}

    def test_measurement_taxonomy(self):
        self.assertEqual(relatedness('healthcare_accessibility','education_accessibility'),(1.,1.))
        self.assertEqual(relatedness('digital_total_cell_intensity','digital_modernization_tmi'),(1.,0.))
        self.assertEqual(relatedness('food_accessibility','population_count'),(0.,0.))
        self.assertEqual(len(MEASUREMENT_ATTRIBUTES),10)

    def test_source_only_model_and_moments(self):
        cal=MeasurementAwareScaleRouter(self.vec,self.stats)
        self.assertGreaterEqual(cal.alpha,0.)
        self.assertLessEqual(cal.alpha,1.)
        mean,std=cal.predict('healthcare_accessibility')
        self.assertTrue(float('-inf')<mean<float('inf'))
        self.assertGreater(std,0.)
        self.assertAlmostEqual(sum(cal.components('healthcare_accessibility')['measurement_weights']),1.)
        with self.assertRaises(ValueError):cal.predict('education_accessibility')

    def test_inner_pseudo_target_exclusion(self):
        sources=['food_accessibility','nighttime_light_intensity']
        a=moment_estimate('education_accessibility',sources,self.vec,self.stats,measurement=True)
        changed=dict(self.stats);changed['education_accessibility']=(9999.,9999.)
        b=moment_estimate('education_accessibility',sources,self.vec,changed,measurement=True)
        self.assertAlmostEqual(a[0],b[0])
        self.assertAlmostEqual(a[1],b[1])

if __name__=='__main__':unittest.main()
