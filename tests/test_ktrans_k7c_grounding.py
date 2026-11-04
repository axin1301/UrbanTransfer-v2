import unittest
import numpy as np
from PIL import Image
from scripts.run_ktrans_k7c_grounding import transform_image,jobs_for
from scripts.diagnose_ktrans_k7c_grounding import paired,summarize,aggregate
from scripts.run_ktrans_k7_vlm_pilot import CUES

class K7CGroundingTests(unittest.TestCase):
    def setUp(self):
        x=np.arange(16*16*3,dtype=np.uint8).reshape(16,16,3)
        self.image=Image.fromarray(x)
        self.tile={'tile_id':'a','city':'City','image_path':'sample.png',
                   'image_name':'sample.png'}
        self.task='education_accessibility'

    def test_gray_eliminates_spatial_information(self):
        a=np.asarray(transform_image(self.image,'blank','a'))
        self.assertEqual(a.shape,(16,16,3))
        self.assertEqual(len(np.unique(a.reshape(-1,3),axis=0)),1)

    def test_shuffle_is_deterministic_and_preserves_pixel_multiset(self):
        original=np.asarray(self.image)
        a=np.asarray(transform_image(self.image,'patch_shuffle','a'))
        b=np.asarray(transform_image(self.image,'patch_shuffle','a'))
        np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(a,original))
        np.testing.assert_array_equal(np.sort(a.reshape(-1,3),axis=0),
                                      np.sort(original.reshape(-1,3),axis=0))

    def test_jobs_unique_and_factorial(self):
        jobs=jobs_for([self.tile],(self.task,),'test',262144)
        self.assertEqual(len(jobs),2)
        self.assertEqual(len(set(j[0] for j in jobs)),2)
        self.assertEqual({j[3] for j in jobs},{'blank','patch_shuffle'})

    def test_paired_missing_cues_not_imputed(self):
        original={'visual_cues':{c:2 for c in CUES},
                  'evidence_relevance':{c:3 for c in CUES},
                  'evidence_confidence':{c:1 for c in CUES}}
        modified={'visual_cues':{c:None for c in CUES},
                  'evidence_relevance':{c:2 for c in CUES},
                  'evidence_confidence':{c:None for c in CUES}}
        result=paired(original,modified)
        self.assertEqual(result['visual_cues_matched'],0)
        self.assertTrue(np.isnan(result['visual_cues_mean_abs_change']))
        self.assertAlmostEqual(result['evidence_relevance_mean_abs_change'],1.)

    def test_missing_intervention_tracked(self):
        p={'scene':'original','proxy_reasoning':'some','limitations':'none',
           'visual_cues':{c:2 for c in CUES},
           'evidence_relevance':{c:2 for c in CUES},
           'evidence_confidence':{c:2 for c in CUES}}
        originals={(self.tile['tile_id'],self.task):{'parsed':p}}
        rec=[{'tile_id':'a','city':'City','task':self.task,
              'condition':'blank','parsed':None}]
        rows=summarize(originals,rec,[self.tile],(self.task,))
        self.assertEqual(len(rows),1)
        summary=aggregate(rows)
        self.assertEqual(summary[0]['paired_valid_n'],0)

if __name__=='__main__':unittest.main()
