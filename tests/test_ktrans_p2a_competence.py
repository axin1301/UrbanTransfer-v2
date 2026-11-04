import unittest
import numpy as np
from urban_transfer.ktrans_k7d import CUES
from urban_transfer.ktrans_p2a import (
    samples_from_rows,task_cue_thresholds,cue_group,summary,
    condition_rows,city_rows,pairwise_rows,cross_city_consistency)

def fixture():
    rows=[]
    for split in ('meta_SCUI','meta_UCUI'):
        for j in range(10):
            high=j>=5
            losses=[1.,3.] if high else [3.,1.]
            for e,loss in zip(('expert_a','expert_b'),losses):
                v=[float(j)/9.]*18
                v[0]=1. if high else 0.
                rows.append({'pseudo_task':'pseudo','inner_partition':split,
                    'tile_id':f'{split}:{j}','city':'new_city' if split=='meta_UCUI' else 'old_city',
                    'expert':e,'label_log1p':2.,'image_path':f'/tmp/{split}_{j}.png',
                    'knowledge_cache_key':f'key:{split}:{j}',
                    'squared_error':loss,'knowledge_scores':v,
                    'knowledge_masks':[1.]*18})
    return rows

class SourceCompetenceP2ATests(unittest.TestCase):
    def test_tiles_not_expert_rows_are_units(self):
        samples=samples_from_rows(fixture(),'outer')
        self.assertEqual(len(samples),20)
        ranked=summary(samples)
        self.assertEqual(len(ranked),4)
        self.assertTrue(all(r['n_tiles']==10 for r in ranked))
        self.assertAlmostEqual(sum(r['fractional_top1_win_rate'] for r in ranked
               if r['inner_partition']=='meta_SCUI'),1.)

    def test_scui_fixed_threshold_and_conditional_effect(self):
        samples=samples_from_rows(fixture(),'outer')
        cut=task_cue_thresholds(samples)
        self.assertAlmostEqual(cut[('pseudo','built_up')],.5)
        row=next(x for x in condition_rows(samples,cut,min_group=5)
            if x['cue']=='built_up' and x['expert']=='expert_a'
            and x['inner_partition']=='meta_UCUI')
        self.assertTrue(row['enough_samples'])
        self.assertEqual((row['n_low'],row['n_high'],row['n_tied']), (5,5,0))
        self.assertAlmostEqual(row['high_minus_low_relative_loss'],-2.)

    def test_fractional_wins_split_ties_per_tile(self):
        rows=fixture()
        # On one tile both experts have exactly the same squared loss.
        for r in rows:
            if r['tile_id']=='meta_SCUI:0':
                r['squared_error']=2.
        summary_rows=summary(samples_from_rows(rows,'outer'))
        scui={r['expert']:r for r in summary_rows
              if r['inner_partition']=='meta_SCUI'}
        self.assertAlmostEqual(scui['expert_a']['fractional_top1_win_rate'],.55)
        self.assertAlmostEqual(scui['expert_b']['fractional_top1_win_rate'],.45)
        self.assertAlmostEqual(sum(r['fractional_top1_win_rate']
                                   for r in scui.values()),1.)

    def test_ucui_cannot_change_thresholds(self):
        rows=fixture()
        initial=task_cue_thresholds(samples_from_rows(rows,'outer'))
        for r in rows:
            if r['inner_partition']=='meta_UCUI':
                r['knowledge_scores'][0]=0.
        result=task_cue_thresholds(samples_from_rows(rows,'outer'))
        self.assertEqual(result,initial)

    def test_missing_vs_zero_and_ties(self):
        rows=fixture()
        for r in rows:
            if r['tile_id']=='meta_SCUI:0':
                r['knowledge_masks'][0]=0.
            if r['tile_id']=='meta_SCUI:1':
                r['knowledge_scores'][0]=.5
        samples=samples_from_rows(rows,'outer')
        threshold=task_cue_thresholds(samples)
        # .5 is a tie only if the frozen SCUI median takes that value.
        self.assertEqual(cue_group(next(s for s in samples
          if s['tile']=='meta_SCUI:0'),'built_up',threshold),'missing')
        self.assertTrue(cue_group(next(s for s in samples
          if s['tile']=='meta_SCUI:1'),'built_up',threshold) in ('low','tied','high'))

    def test_reject_inconsistent_or_missing_expert(self):
        rows=fixture()
        with self.assertRaises(ValueError):
            samples_from_rows(rows[:-1],'outer')
        rows=fixture();rows[1]['knowledge_scores'][0]=.2
        with self.assertRaises(ValueError):
            samples_from_rows(rows,'outer')
        with self.assertRaises(ValueError):
            samples_from_rows([{**r,'expert':'outer'} for r in fixture()],'outer')

    def test_paired_and_city_reports(self):
        samples=samples_from_rows(fixture(),'outer')
        pairs=pairwise_rows(samples)
        self.assertEqual(len(pairs),2)
        self.assertEqual(pairs[0]['n_tiles'],10)
        self.assertAlmostEqual(pairs[0]['expert_a_win_rate_excluding_ties'],.5)
        self.assertEqual(len(city_rows(samples)),4)
        contrast=condition_rows(samples,task_cue_thresholds(samples),min_group=5)
        consistency=cross_city_consistency(contrast,summary(samples))
        row=next(x for x in consistency
                 if x['cue']=='built_up' and x['expert']=='expert_a')
        self.assertTrue(row['evaluable'])
        self.assertTrue(row['same_direction'])

if __name__=='__main__':unittest.main()
