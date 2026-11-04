import json
import tempfile
import unittest
from pathlib import Path
from scripts.cache_ktrans_k7d import unique_jobs,get_pending
from scripts.run_ktrans_k7d import index_knowledge
from urban_transfer.ktrans_k7d import CUES,encode_record,knowledge_vector
from urban_transfer.ktrans_k7d_image_only import (
    IMAGE_ONLY_PROMPT,IMAGE_ONLY_VERSION,parse_image_only)

class K7DImageOnlyV2Tests(unittest.TestCase):
    def setUp(self):
        self.tiles={'rows':[{'tile_id':'one','task':'education_accessibility',
                              'image_path':'/tmp/one.png'}]}
        self.scores={c:2 for c in CUES}

    def test_new_schema_has_no_relevance(self):
        raw=json.dumps({
            'scene':'Urban morphology','visual_cues':self.scores,
            'evidence_confidence':self.scores,
            'limitations':'Functions unknown'})
        parsed=parse_image_only(raw)
        self.assertIsNotNone(parsed)
        self.assertNotIn('evidence_relevance',parsed)
        vector=encode_record(parsed,'image_only')
        self.assertEqual(len(vector),36)
        self.assertEqual(vector[6:12].sum().item(),0.)
        self.assertEqual(vector[24:30].sum().item(),0.)

    def test_forbids_old_relevance_and_fields(self):
        raw={'scene':'Urban','visual_cues':self.scores,
             'evidence_confidence':self.scores,'limitations':'Unknown',
             'evidence_relevance':self.scores}
        self.assertIsNone(parse_image_only(json.dumps(raw)))
        raw.pop('evidence_relevance')
        raw['visual_cues']={'roads':2}
        self.assertIsNone(parse_image_only(json.dumps(raw)))

    def test_only_image_only_identity_changes(self):
        jobs=unique_jobs(self.tiles,'test',262144)
        self.assertEqual(len(jobs),3)
        image=[j for j in jobs if j['condition']=='image_only'][0]
        others=[j for j in jobs if j['condition']!='image_only']
        self.assertEqual(image['version'],IMAGE_ONLY_VERSION)
        self.assertEqual(image['prompt'],IMAGE_ONLY_PROMPT)
        self.assertTrue(all(j['version']!=IMAGE_ONLY_VERSION for j in others))
        self.assertEqual(len({j['key'] for j in jobs}),3)

    def test_cache_ignores_legacy_invalid_image_only(self):
        jobs=unique_jobs(self.tiles,'test',262144)
        v2=[j for j in jobs if j['condition']=='image_only'][0]
        records={'legacy-imageonly-key':{'parsed':None}}
        self.assertIn(v2,get_pending(jobs,records))
        with self.assertRaises(ValueError):
            index_knowledge(jobs,records)

    def test_image_only_ablation_accepts_v2(self):
        v2={'scene':'City','visual_cues':self.scores,
            'evidence_confidence':self.scores,'limitations':'Unknown'}
        text={'visual_cues':{c:None for c in CUES},
              'evidence_relevance':self.scores,
              'evidence_confidence':{c:None for c in CUES}}
        image_task={'visual_cues':self.scores,'evidence_relevance':self.scores,
                    'evidence_confidence':self.scores}
        for variant in ('image_only','image_task','text_only','k5a'):
            self.assertEqual(len(knowledge_vector(image_task,v2,text,variant)),36)

if __name__=='__main__':unittest.main()
