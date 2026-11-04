import json
import tempfile
import unittest
from pathlib import Path
from scripts.run_ktrans_k7b_controls import (
    CUES,build_prompt,parse,jobs_for,load_cached,read_tile_manifest)
from scripts.diagnose_ktrans_k7b_controls import compute

class K7BControlsTests(unittest.TestCase):
    def setUp(self):
        self.scored={c:2 for c in CUES}
        self.empty={c:None for c in CUES}

    def _obj(self,mode):
        return {'scene':'Some built up structures',
                'visual_cues':self.empty if mode=='text_only' else self.scored,
                'evidence_relevance':self.empty if mode=='image_only' else self.scored,
                'evidence_confidence':self.empty if mode=='text_only' else self.scored,
                'proxy_reasoning':'visible streets can indirectly inform urban morphology',
                'limitations':'No direct indicator measurement'}

    def test_parse_all_conditions(self):
        for mode in ('image_task','text_only','image_only'):
            self.assertIsNotNone(parse(json.dumps(self._obj(mode)),mode))

    def test_reject_false_visual_claims_without_image(self):
        item=self._obj('text_only')
        item['visual_cues']=self.scored
        self.assertIsNone(parse(json.dumps(item),'text_only'))

    def test_reject_task_relevance_without_task(self):
        item=self._obj('image_only')
        item['evidence_relevance']=self.scored
        self.assertIsNone(parse(json.dumps(item),'image_only'))

    def test_task_prompt_contains_task_specific_definition(self):
        d={'education_accessibility':'distance-decay school access',
           'nighttime_light_intensity':'nighttime luminosity'}
        image=build_prompt('education_accessibility','image_task',d)
        text=build_prompt('education_accessibility','text_only',d)
        self.assertIn('distance-decay school access',image)
        self.assertIn('distance-decay school access',text)
        self.assertIn('NO image',text)

    def test_exact_factorial_jobs_and_cache(self):
        tiles=[{'tile_id':'a','city':'A','image_name':'a.png','image_path':'a.png'},
               {'tile_id':'b','city':'B','image_name':'b.png','image_path':'b.png'}]
        tasks=('education_accessibility','nighttime_light_intensity')
        jobs=jobs_for(tiles,tasks,'test',262144)
        self.assertEqual(len(jobs),2*(1+len(tasks))+len(tasks))
        self.assertEqual(len(set(j[0] for j in jobs)),len(jobs))
        self.assertEqual(sum(j[3]=='text_only' for j in jobs),2)

    def test_analyzer_separates_text_prior_and_image_shift(self):
        def row(mode,task,tile=None,value=2):
            obj=self._obj(mode)
            if mode!='image_only':
                obj['evidence_relevance']={c:value for c in CUES}
            return {'condition':mode,'task':task,'tile_id':tile,
                    'city':'A','parsed':obj}
        task='education_accessibility'
        records=[row('text_only',task,value=1),
                 row('image_task',task,'a',2),
                 row('image_task',task,'b',3),
                 row('image_only','image_only','a'),
                 row('image_only','image_only','b')]
        tasks,tiles=compute(records,(task,))
        self.assertEqual(tasks[0]['valid_image_task'],2)
        self.assertAlmostEqual(tasks[0]['mean_abs_image_vs_text_relevance'],1.5)
        self.assertAlmostEqual(tasks[0]['mean_within_task_across_image_std'],.5)
        self.assertEqual(len(tiles),2)

if __name__=='__main__':unittest.main()
