import unittest
import numpy as np
from urban_transfer.ktrans_p2b1 import (
    aligned_samples,relative_target,sample_features,matrix,
    eval_predictions,source_lopo,summarize_source_lopo)

TASKS=('a','b','c')

def fixture():
    rows=[]
    for t in TASKS:
        for split in ('meta_SCUI','meta_UCUI'):
            for j in range(5):
                experts=tuple(s for s in TASKS if s!=t)
                label=2.
                for k,expert in enumerate(experts):
                    pred=label+(0.2 if (j+k)%2==0 else 0.6)
                    vals=[.5]*18
                    vals[0]=j/4
                    rows.append({'pseudo_task':t,'inner_partition':split,
                        'tile_id':f'{t}:{split}:{j}','city':f'{split}:city',
                        'expert':expert,'label_log1p':label,'prediction_log1p':pred,
                        'squared_error':(pred-label)**2,
                        'knowledge_scores':vals,'knowledge_masks':[1.]*18,
                        'image_path':f'/tmp/{t}{split}{j}.png',
                        'knowledge_cache_key':f'key:{t}{split}{j}'})
    return rows

def texts():
    return {t:[.2+index*.2]*6 for index,t in enumerate(TASKS)}

class P2B1Tests(unittest.TestCase):
    def test_source_tile_alignment_and_loss_consistency(self):
        samples=aligned_samples(fixture(),'outer',TASKS)
        self.assertEqual(len(samples),30)
        self.assertTrue(all(len(s['experts'])==2 for s in samples))
        for s in samples:
            np.testing.assert_allclose((s['pred']-s['target'])**2,s['loss'])
            self.assertAlmostEqual(float(relative_target(s).mean()),0.)

    def test_reject_outer_target_or_own_expert(self):
        rows=fixture()
        rows[0]['expert']='outer'
        with self.assertRaises(ValueError):
            aligned_samples(rows,'outer',TASKS)
        rows=fixture()
        rows[0]['prediction_log1p']+=1.
        with self.assertRaises(ValueError):
            aligned_samples(rows,'outer',TASKS)

    def test_global_expert_features_not_fold_position(self):
        samples=aligned_samples(fixture(),'outer',TASKS)
        a=sample_features(samples[0],samples[0]['experts'][0],TASKS,texts(),'task_expert')
        b=sample_features(samples[0],samples[0]['experts'][1],TASKS,texts(),'task_expert')
        self.assertEqual(a.shape,(1+3+3*6,))
        self.assertNotEqual(np.argmax(a[1:4]),np.argmax(b[1:4]))
        for mode in ('cue_expert','task_cue_expert','permuted_task_cue_expert'):
            values,_,sizes=matrix(samples,TASKS,texts(),mode,targets=False)
            self.assertEqual(len(values),60)
            self.assertEqual(sizes,[2]*30)

    def test_soft_mixture_and_pairwise_correct(self):
        samples=aligned_samples(fixture(),'outer',TASKS)[:1]
        # Perfect ranking: lower true loss gets a lower competence score.
        metric=eval_predictions(samples,[samples[0]['loss']],.5)[0]
        self.assertEqual(metric['pairwise_n'],1)
        self.assertAlmostEqual(metric['pairwise_correct_sum'],1.)
        self.assertAlmostEqual(metric['top1_fractional_correct'],1.)
        self.assertGreaterEqual(metric['soft_mix_squared_error'],0.)

    def test_source_task_holdout_and_ucui_not_in_training(self):
        samples=aligned_samples(fixture(),'outer',TASKS)
        out=source_lopo(samples,TASKS,texts(),alpha=10.,temperature=.5,seed=42)
        self.assertEqual(len(out),30*4)
        tasks,macro=summarize_source_lopo(out)
        self.assertEqual(len(tasks),len(TASKS)*2*4)
        self.assertEqual(len(macro),2*4)
        self.assertTrue(all(r['n_source_pseudo_tasks']==3 for r in macro))
        # Disturbing UCUI true source labels changes only evaluation
        # metrics, never source SCUI labels or fitted ridge coefficients.
        mutated=aligned_samples(fixture(),'outer',TASKS)
        for s in mutated:
            if s['split']=='meta_UCUI':
                s['loss']=s['loss']+10.
                s['relative_loss']=s['loss']-s['loss'].mean()
        modified=source_lopo(mutated,TASKS,texts(),alpha=10.,temperature=.5,seed=42)
        scui=[r for r in out if r['inner_partition']=='meta_SCUI']
        new_scui=[r for r in modified if r['inner_partition']=='meta_SCUI']
        self.assertEqual(scui,new_scui)

if __name__=='__main__':unittest.main()
