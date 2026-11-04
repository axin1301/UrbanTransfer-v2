import unittest
import numpy as np
from urban_transfer.ktrans_p2b1 import aligned_samples,source_lopo
from urban_transfer.ktrans_p2d_vlm import (
    MODES,vlm_components,features,ridge_predict,
    permute_task_specific_knowledge,run_cross_task,run_same_task,
    summarize_same_task)

TASKS=('a','b','c','d','e')
def synthetic():
    rows=[]
    for task_i,task in enumerate(TASKS):
        for split in ('meta_SCUI','meta_UCUI'):
            for city in (('seen1','seen2') if split=='meta_SCUI' else ('new1','new2')):
                for j in range(6):
                    tile=f'{task}:{split}:{city}:{j}'
                    signal=float(j%3)/3.
                    scores=[0.5]*18
                    scores[0]=signal
                    scores[6]=float((j+task_i)%4)/3.
                    scores[12]=1.
                    masks=[1.]*18
                    if j==0:masks[6]=0.;scores[6]=0.
                    for expert in TASKS:
                        if expert==task:continue
                        err=float((TASKS.index(expert)+j+task_i)%3)*.3+.05
                        rows.append({'pseudo_task':task,
                            'inner_partition':split,
                            'tile_id':tile,'city':city,'expert':expert,
                            'prediction_log1p':2.+err,'label_log1p':2.,
                            'squared_error':err**2,
                            'knowledge_scores':scores,
                            'knowledge_masks':masks,
                            'image_path':f'/tmp/{tile}.png',
                            'knowledge_cache_key':tile})
    return aligned_samples(rows,'outer',TASKS)

def texts():
    return {t:[(index+1)/6.]*6 for index,t in enumerate(TASKS)}

class P2DVLMTests(unittest.TestCase):
    def test_direct_relevance_is_actual_task_conditioned_field(self):
        sample=synthetic()[1]
        one=features(sample,sample['experts'][0],TASKS,texts(),'vlm_task_relevance')
        altered={**sample,'visual':sample['visual'].copy()}
        altered['visual'][1,0]=1.-altered['visual'][1,0]
        other=features(altered,sample['experts'][0],TASKS,texts(),'vlm_task_relevance')
        self.assertFalse(np.array_equal(one,other))
        legacy=features(sample,sample['experts'][0],TASKS,texts(),'task_expert')
        legacy_again=features(altered,sample['experts'][0],TASKS,texts(),'task_expert')
        np.testing.assert_array_equal(legacy,legacy_again)

    def test_null_task_relevance_mask_zero_and_residual(self):
        sample=synthetic()[0]
        _,relevance,_,conditional,residual,prior,masks=vlm_components(sample,texts())
        self.assertEqual(masks[1,0],0.)
        self.assertEqual(relevance[0],0.)
        self.assertEqual(conditional[0],0.)
        self.assertEqual(residual[0],0.)
        self.assertEqual(prior[0],texts()[sample['task']][0])

    def test_permutation_changes_only_task_relevance_not_visual_confidence(self):
        group=synthetic()
        alt=permute_task_specific_knowledge(group,np.random.default_rng(42))
        self.assertEqual(len(group),len(alt))
        for a,b in zip(group,alt):
            self.assertEqual((a['task'],a['city'],a['tile'],a['experts']),
                             (b['task'],b['city'],b['tile'],b['experts']))
            np.testing.assert_array_equal(a['visual'][0],b['visual'][0])
            np.testing.assert_array_equal(a['visual'][2],b['visual'][2])
            np.testing.assert_array_equal(a['mask'][0],b['mask'][0])
            np.testing.assert_array_equal(a['mask'][2],b['mask'][2])
            np.testing.assert_array_equal(a['loss'],b['loss'])
        self.assertTrue(any(not np.array_equal(a['visual'][1],b['visual'][1])
                            for a,b in zip(group,alt)))
        for task in TASKS:
            for split in ('meta_SCUI','meta_UCUI'):
                original=sorted(tuple(s['visual'][1]) for s in group
                                if s['task']==task and s['split']==split)
                shuffled=sorted(tuple(s['visual'][1]) for s in alt
                                if s['task']==task and s['split']==split)
                self.assertEqual(original,shuffled)

    def test_fixed_baseline_matches_p2b1_scores(self):
        samples=synthetic()
        train=[x for x in samples if x['task']!='a' and x['split']=='meta_SCUI']
        test=[x for x in samples if x['task']=='a' and x['split']=='meta_UCUI']
        from urban_transfer.ktrans_p2b1 import ridge_predict as old
        for mode in ('task_expert','task_cue_expert'):
            a=old(train,test,TASKS,texts(),mode,10.)
            b=ridge_predict(train,test,TASKS,texts(),mode,10.)
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

    def test_complete_cross_and_same_task_source_only(self):
        samples=synthetic()
        cross=run_cross_task(samples,TASKS,texts(),alpha=10.,temperature=.5)
        same=run_same_task(samples,TASKS,texts(),alpha=10.,temperature=.5)
        self.assertEqual(len(cross),len(samples)*len(MODES))
        self.assertEqual(len(same),sum(s['split']=='meta_UCUI' for s in samples)*len(MODES))
        from urban_transfer.ktrans_p2b1 import summarize_source_lopo
        tasks,macro=summarize_source_lopo(cross)
        self.assertEqual(len(macro),len(MODES)*2)
        _,same_macro=summarize_same_task(same)
        self.assertEqual(len(same_macro),len(MODES))
        self.assertTrue(all(np.isfinite(r['macro_pairwise_accuracy']) for r in macro))

    def test_ucui_labels_cannot_change_fitted_mixture_predictions(self):
        samples=synthetic()
        prior=run_cross_task(samples,TASKS,texts(),alpha=10.,temperature=.5)
        altered=[]
        for s in samples:
            if s['split']=='meta_UCUI':
                alter={**s,'target':s['target']+1.}
                alter['loss']=(s['pred']-alter['target'])**2
            else:alter=s
            altered.append(alter)
        changed=run_cross_task(altered,TASKS,texts(),alpha=10.,temperature=.5)
        self.assertEqual(len(prior),len(changed))
        for a,b in zip(prior,changed):
            self.assertAlmostEqual(a['soft_mix_prediction'],b['soft_mix_prediction'])

if __name__=='__main__':unittest.main()
