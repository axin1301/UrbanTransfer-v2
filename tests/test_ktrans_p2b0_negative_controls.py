import unittest
import numpy as np
from urban_transfer.ktrans_p2a import samples_from_rows,task_cue_thresholds
from urban_transfer.ktrans_p2b0 import (
    cue_permutation,city_bootstrap,consistency_report,
    run_negative_controls,effect_by_task_cue)

def fixture():
    rows=[]
    for split in ('meta_SCUI','meta_UCUI'):
        for city in ('city_a','city_b'):
            for j in range(12):
                high=j>=6
                for expert,loss in zip(('expert_a','expert_b'),
                                       ([1.,3.] if high else [3.,1.])):
                    vals=[.5]*18
                    vals[0]=1. if high else 0.
                    rows.append({'pseudo_task':'pseudo','inner_partition':split,
                        'city':city,'tile_id':split+':'+city+':'+str(j),
                        'expert':expert,'squared_error':loss,'label_log1p':2.,
                        'knowledge_scores':vals,'knowledge_masks':[1.]*18,
                        'image_path':'/tmp/'+split+city+str(j)+'.png',
                        'knowledge_cache_key':split+city+str(j)})
    return samples_from_rows(rows,'outer')

class P2B0ControlsTest(unittest.TestCase):
    def test_shuffle_preserves_losses_task_and_city(self):
        src=fixture()
        dst,stats=cue_permutation(src,np.random.default_rng(2))
        self.assertEqual(len(src),len(dst))
        self.assertGreater(stats['n_evidence_reassigned'],0)
        for before,after in zip(src,dst):
            self.assertEqual((before['task'],before['split'],before['city'],before['tile']),
                             (after['task'],after['split'],after['city'],after['tile']))
            np.testing.assert_array_equal(before['loss'],after['loss'])
        self.assertTrue(any(not np.array_equal(a['visual'],b['visual'])
                            for a,b in zip(src,dst)))

    def test_city_stratified_shuffle_retains_city_evidence_multiset(self):
        src=fixture()
        dst,_=cue_permutation(src,np.random.default_rng(3),within_city=True)
        for split in ('meta_SCUI','meta_UCUI'):
            for city in ('city_a','city_b'):
                def cue_hist(items):
                    return sorted(float(s['visual'][0,0]) for s in items
                        if s['split']==split and s['city']==city)
                self.assertEqual(cue_hist(src),cue_hist(dst))

    def test_bootstrap_samples_whole_cities(self):
        src=fixture()
        alt=city_bootstrap(src,np.random.default_rng(4))
        self.assertEqual(len(alt),len(src))
        for split in ('meta_SCUI','meta_UCUI'):
            city_counts={c:sum(s['city']==c and s['split']==split for s in alt)
                         for c in ('city_a','city_b')}
            self.assertEqual(sum(city_counts.values()),24)
            self.assertTrue(all(n%12==0 for n in city_counts.values()))

    def test_source_scui_cut_is_frozen_under_ucui_shuffles(self):
        src=fixture()
        thresholds=task_cue_thresholds(src)
        shuffled,_=cue_permutation(src,np.random.default_rng(9))
        self.assertEqual(thresholds,task_cue_thresholds(shuffled))
        observed,detail=consistency_report(src,thresholds,min_group=5)
        self.assertGreater(observed['n_evaluable'],0)
        self.assertAlmostEqual(observed['agreement_rate'],1.)
        self.assertTrue(effect_by_task_cue(detail))

    def test_repeated_controls_reproducible_and_not_relabelled(self):
        src=fixture()
        a=run_negative_controls(src,seed=42,repeats=3,min_group=5)
        b=run_negative_controls(src,seed=42,repeats=3,min_group=5)
        self.assertEqual(a[0],b[0])
        # A shuffled cohort may have no evaluable high/low contrasts:
        # both runs legitimately report NaN; NaN != NaN for list/dict
        # equality, so compare numeric metrics with equal_nan=True.
        self.assertEqual(len(a[2]),len(b[2]))
        for first,second in zip(a[2],b[2]):
            self.assertEqual(first.keys(),second.keys())
            for key in first:
                if isinstance(first[key],float):
                    np.testing.assert_allclose(first[key],second[key],
                                               rtol=0,atol=0,equal_nan=True)
                else:
                    self.assertEqual(first[key],second[key])
        self.assertTrue(any(
            np.isnan(r['agreement_rate']) for r in a[2]
        ),'Fixture should exercise undefined agreement in shuffled cohorts')
        self.assertEqual(len(a[2]),9)
        self.assertEqual(len(a[3]),3)
        self.assertTrue(all(0<r['descriptive_tail_fraction_ge_observed']<=1
                            for r in a[3]))

if __name__=='__main__':unittest.main()
