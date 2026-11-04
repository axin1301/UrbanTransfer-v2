import json
import tempfile
import unittest
from pathlib import Path
from scripts.cache_ktrans_k7d import merge_records,unique_jobs,get_pending
from scripts.repair_ktrans_k7d_image_only import recover,write_repair
from urban_transfer.ktrans_k7d_recovery import project_image_only_v2
from urban_transfer.ktrans_k7d_image_only import IMAGE_ONLY_VERSION
from urban_transfer.ktrans_k7d import CUES,encode_record

class K7DCacheRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.scores={cue:2 for cue in CUES}
        self.raw={'scene':'Overhead image with roads and vegetation',
                  'visual_cues':self.scores,'evidence_confidence':self.scores,
                  'limitations':'Cannot infer ground truth'}
        self.manifest={'rows':[{'tile_id':'tileA','task':'education_accessibility',
                                 'image_path':'/tmp/a.png'}]}
        self.job=[j for j in unique_jobs(self.manifest,'test-model',262144)
                  if j['condition']=='image_only'][0]

    def test_strict_v2_passes_without_projection(self):
        parsed,extras,why=project_image_only_v2(json.dumps(self.raw))
        self.assertEqual(why,'strict')
        self.assertEqual(extras,())
        self.assertEqual(parsed['visual_cues']['roads'],2)

    def test_discards_only_named_extra_fields_with_audit(self):
        raw={**self.raw,'evidence_relevance':self.scores,
             'proxy_reasoning':'Speculative task-specific statement'}
        parsed,extras,why=project_image_only_v2(json.dumps(raw))
        self.assertEqual(why,'projected_extra_fields')
        self.assertEqual(set(extras),{'evidence_relevance','proxy_reasoning'})
        self.assertNotIn('evidence_relevance',parsed)
        x=encode_record(parsed,'image_only')
        self.assertEqual(float(x[6:12].sum()),0.)
        self.assertEqual(float(x[24:30].sum()),0.)

    def test_invalid_scores_are_not_fabricated(self):
        raw={**self.raw,'visual_cues':{'roads':3}}
        parsed,extras,why=project_image_only_v2(json.dumps(raw))
        self.assertIsNone(parsed)
        self.assertEqual(why,'unexpected_schema')
        self.assertIsNone(project_image_only_v2('{"truncated":')[0])

    def test_recovery_appends_same_key_preserves_raw(self):
        raw={**self.raw,'evidence_relevance':self.scores}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'evidence.jsonl'
            row={**self.job,'raw':json.dumps(raw),'parsed':None}
            path.write_text(json.dumps(row)+'\n')
            add,audit,counts,total=recover(self.manifest,path,Path(tmp)/'none',
                                            'test-model',262144)
            self.assertEqual(total,1)
            self.assertEqual(len(add),1)
            self.assertEqual(counts['projected_extra_fields'],1)
            self.assertEqual(add[0]['key'],row['key'])
            self.assertEqual(add[0]['raw'],row['raw'])
            self.assertEqual(add[0]['discarded_extra_fields'],['evidence_relevance'])
            write_repair(path,add)
            cache=merge_records([path])
            self.assertEqual(len(get_pending([self.job],cache)),0)

    def test_merge_records_keeps_valid_if_later_attempt_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.jsonl'
            good={'key':'k','parsed':{'scene':'ok'}}
            bad={'key':'k','parsed':None}
            path.write_text(json.dumps(good)+'\n'+json.dumps(bad)+'\n')
            self.assertIsNotNone(merge_records([path])['k']['parsed'])

if __name__=='__main__':unittest.main()
