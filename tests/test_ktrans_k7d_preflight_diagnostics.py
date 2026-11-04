import json
import unittest
from scripts.cache_ktrans_k7d import unique_jobs
from scripts.diagnose_ktrans_k7d_image_only import inspect
from urban_transfer.ktrans_k7d import CUES

class K7DImageOnlyDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.manifest={'rows':[{'tile_id':'tile','task':'education_accessibility',
                                'image_path':'/tmp/a.png'}]}
        self.job=[j for j in unique_jobs(self.manifest,'test',262144)
                  if j['condition']=='image_only'][0]

    def test_missing_job_classified(self):
        rows,counts=inspect(self.manifest,{},'test',262144)
        self.assertEqual(counts['never_generated'],1)

    def test_valid_and_projectable_are_distinct(self):
        fields={c:2 for c in CUES}
        response={'scene':'roads','visual_cues':fields,
                  'evidence_confidence':fields,'limitations':'unknown',
                  'evidence_relevance':fields}
        rows,counts=inspect(self.manifest,{
            self.job['key']:{**self.job,'parsed':None,'raw':json.dumps(response)}
        },'test',262144)
        self.assertEqual(counts['projectable'],1)
        self.assertIn('evidence_relevance',rows[0]['discarded_extra_fields'])

    def test_truncated_reported_without_repair(self):
        rows,counts=inspect(self.manifest,{
            self.job['key']:{**self.job,'parsed':None,
                             'raw':'{"scene":"roads"'}
        },'test',262144)
        self.assertEqual(counts['invalid_json'],1)
        self.assertGreater(rows[0]['raw_chars'],0)

if __name__=='__main__':unittest.main()
