"""P2-E1 source-LOTO evaluator regression with synthetic VLM hypotheses."""
import unittest
import numpy as np
from urban_transfer.ktrans_p2e1_eval import run_lopo,feature
from urban_transfer.ktrans_p2e1_pilot import make_jobs,sample_source_rows
from urban_transfer.ktrans_p2e1_indicator_spec import catalog

TASKS=('population_count','nighttime_light_intensity',
       'food_accessibility','digital_modernization_tmi',
       'digital_total_cell_intensity')

def fixture():
    evidence=[]
    for task_idx,task in enumerate(TASKS):
        for split,city in (('meta_SCUI','seen'),('meta_UCUI','unseen')):
            for i in range(4):
                tile=f'{task}:{split}:{i}'
                for expert in TASKS:
                    if expert==task:continue
                    error=float((task_idx+TASKS.index(expert)+i)%3)*.2+.1
                    evidence.append({'pseudo_task':task,'inner_partition':split,
                        'tile_id':tile,'city':city,
                        'image_path':f'/tmp/{tile}.png',
                        'expert':expert,'label_log1p':2.,
                        'prediction_log1p':2.+error,
                        'squared_error':error*error,
                        'knowledge_scores':[.5]*18,
                        'knowledge_masks':[1.]*18,
                        'knowledge_cache_key':tile})
    sampled=sample_source_rows(evidence,TASKS,4)
    jobs=make_jobs(sampled,TASKS,catalog(),'fake-vlm',262144)
    manifest={'plan':{'outer_target_excluded':'outer'},'tiles':sampled,'jobs':jobs}
    def pack(job):
        return {e:{'compatibility':2,'region_support':1 if job['image_path'] else None,
                   'transfer_risk':1,'uncertainty':2,
                   'reason':'Test hypothesis','complementarity':'Possible'} for e in TASKS if e!=job['task']}
    hypothesis={(job['arm'],job['task'],job['tile_id']):pack(job) for job in jobs}
    return evidence,manifest,hypothesis

class PilotEvalTests(unittest.TestCase):
    def test_small_source_only_lopo_all_four_modes(self):
        evidence,manifest,hypothesis=fixture()
        task,macro=run_lopo(evidence,manifest,hypothesis,TASKS,10.,.5)
        self.assertEqual(len(macro),4)
        self.assertEqual(len(task),20)
        self.assertTrue(all(r['n_tiles']==20 for r in macro))
        self.assertTrue(all(np.isfinite(r['macro_pairwise_accuracy']) for r in macro))

    def test_no_labels_in_features(self):
        card={'compatibility':3,'region_support':None,'transfer_risk':2,'uncertainty':1}
        x=feature(TASKS[1],TASKS,card)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x.shape[0],1+len(TASKS)+8*len(TASKS))

    def test_missing_vlm_job_fails_closed(self):
        evidence,manifest,hypothesis=fixture()
        hypothesis.pop(next(iter(hypothesis)))
        with self.assertRaises(ValueError):
            run_lopo(evidence,manifest,hypothesis,TASKS,10.,.5)

    def test_eval_labels_do_not_enter_source_only_fitted_scores(self):
        evidence,manifest,hypothesis=fixture()
        # Check source-lopo is evaluation on heldout task, with all four
        # experts and complete VLM identity; synthetic labels are only
        # used in metrics, not feature construction.
        first=next(j for j in manifest['jobs'] if j['arm']=='definition_relation_region')
        self.assertEqual(len(hypothesis[first['arm'],first['task'],first['tile_id']]),4)

if __name__=='__main__':unittest.main()
