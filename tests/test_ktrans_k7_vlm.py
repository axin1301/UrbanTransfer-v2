import json
import tempfile
import unittest
from pathlib import Path
from scripts.run_ktrans_k7_vlm_pilot import (
 CUES,prompt,parse_output,cache_key,select_tiles)

class VlmPilotTests(unittest.TestCase):
    def setUp(self):
        self.empty={c:None for c in CUES}
        self.values={c:2 for c in CUES}

    def test_parse_valid_json(self):
        record={'scene':'Visible streets','visual_cues':self.values,
                'task_relevance':self.empty,'limitations':'No numerical label visible'}
        x=parse_output(json.dumps(record))
        self.assertEqual(x['visual_cues']['roads'],2)

    def test_reject_out_of_bounds(self):
        bad=dict(self.values,built_up=4)
        self.assertIsNone(parse_output(json.dumps({
            'visual_cues':bad,'task_relevance':self.empty})))

    def test_reject_unparseable(self):
        self.assertIsNone(parse_output('unstructured free text'))

    def test_image_only_vs_conditioned_prompt(self):
        descriptions={'education_accessibility':'school access definition'}
        a=prompt('image_only',descriptions)
        b=prompt('education_accessibility',descriptions)
        self.assertNotEqual(a,b)
        self.assertIn('school access definition',b)
        self.assertNotIn('school access definition',a)

    def test_cache_key_distinguishes_tasks_and_prompt(self):
        first=cache_key('a','image_only','Qwen3', 'describe',262144)
        second=cache_key('a','education_accessibility','Qwen3','describe',262144)
        third=cache_key('a','image_only','Qwen3','describe better',262144)
        self.assertNotEqual(first,second)
        self.assertNotEqual(first,third)

if __name__=='__main__':unittest.main()
