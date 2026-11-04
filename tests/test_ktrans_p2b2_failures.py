import unittest
import numpy as np
from urban_transfer.ktrans_p2b1 import aligned_samples
from urban_transfer.ktrans_p2b2 import (
    source_partition_checks,same_task_predictions,cross_task_predictions,
    route_metrics,expert_structure)

TASKS=('task_a','task_b','task_c')

def fixture():
    rows=[]
    for task in TASKS:
        experts=[e for e in TASKS if e!=task]
        for split in ('meta_SCUI','meta_UCUI'):
            for city in (('seen1','seen2') if split=='meta_SCUI' else ('unseen1','unseen2')):
                for idx in range(4):
                    tile=f'{task}:{split}:{city}:{idx}'
                    value=2.
                    for k,expert in enumerate(experts):
                        pred=2.+(.2 if (idx+k)%2==0 else .6)
                        cues=[.5]*18
                        cues[0]=idx/3
                        rows.append({'pseudo_task':task,'inner_partition':split,
                            'tile_id':tile,'city':city,
                            'expert':expert,'label_log1p':value,
                            'prediction_log1p':pred,'squared_error':(pred-value)**2,
                            'knowledge_scores':cues,'knowledge_masks':[1.]*18,
                            'image_path':'/tmp/'+tile+'.png',
                            'knowledge_cache_key':tile})
    return aligned_samples(rows,'outer',TASKS)

def texts():
    return {name:[(i+1)/4]*6 for i,name in enumerate(TASKS)}

class P2B2Tests(unittest.TestCase):
    def test_partition_city_and_source_expert_identity(self):
        s=fixture()
        source_partition_checks(s,TASKS)
        contaminated=[dict(x) for x in s]
        for row in contaminated:
            if row['split']=='meta_UCUI' and row['task']=='task_a':
                row['city']='seen1'
        with self.assertRaises(ValueError):
            source_partition_checks(contaminated,TASKS)

    def test_same_and_cross_test_exact_same_ucui_tiles(self):
        s=fixture()
        same=same_task_predictions(s,TASKS,texts(),alpha=10.,temperature=.5)
        cross=cross_task_predictions(s,TASKS,texts(),alpha=10.,temperature=.5)
        each=3*2*4*4
        self.assertEqual(len(same),each)
        self.assertEqual(len(cross),each)
        per_task,macro=route_metrics(same+cross)
        self.assertEqual(len(per_task),3*2*4)
        self.assertEqual(len(macro),2*4)
        self.assertEqual({r['route'] for r in macro},
                         {'same_task_scui_to_ucui','other_tasks_scui_to_ucui'})
        self.assertTrue(all(np.isfinite(r['macro_pairwise_accuracy']) for r in macro))

    def test_strict_ucui_not_in_fit(self):
        original=fixture()
        before=same_task_predictions(original,TASKS,texts())
        # Changed UCUI labels should change evaluation losses, but must
        # not change frozen SCUI-only weights (soft-mix predictions).
        changed=fixture()
        for row in changed:
            if row['split']=='meta_UCUI':
                row['target']+=1.
                row['loss']=(row['pred']-row['target'])**2
                row['relative_loss']=row['loss']-row['loss'].mean()
        after=same_task_predictions(changed,TASKS,texts())
        self.assertEqual(len(before),len(after))
        for a,b in zip(before,after):
            self.assertAlmostEqual(a['soft_mix_prediction'],b['soft_mix_prediction'])
            self.assertNotEqual(a['target_log1p'],b['target_log1p'])

    def test_expert_covariance_and_scui_fixed_baseline(self):
        detail,pairs,rankings=expert_structure(fixture(),TASKS,.05)
        self.assertEqual(len(detail),6)
        self.assertEqual(len(pairs),6)
        self.assertEqual(len(rankings),3)
        self.assertTrue(all(r['n_tiles']==8 for r in detail))
        self.assertTrue(all(r['n_informative_tile_spans']>0 for r in detail))
        self.assertTrue(all(r['scui_fixed_best_expert'] in TASKS for r in detail))
        self.assertTrue(all(np.isfinite(r['mean_error_product']) for r in pairs))

    def test_strict_same_ucui_comparison_identity(self):
        s=fixture()
        same=same_task_predictions(s,TASKS,texts())
        cross=cross_task_predictions(s,TASKS,texts())
        corrupted=[dict(row) for row in cross]
        corrupted[0]['tile_id']='wrong_tile'
        with self.assertRaises(ValueError):
            route_metrics(same+corrupted)

if __name__=='__main__':unittest.main()
