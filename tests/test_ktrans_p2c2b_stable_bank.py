import unittest
import numpy as np
from urban_transfer.ktrans_p2b1 import aligned_samples
from urban_transfer.ktrans_p2c2b_stable_bank import (
    prepare_records,fit_bank,predict_bank_row,export_stable_evidence,bank_report,
    save_bank,load_bank)

TASKS=('a','b','c','d','e')

def evidence_fixture():
    sources=[]
    old=[]
    for i,task in enumerate(TASKS):
        for city in ('seen1','seen2','unseen1'):
            split='meta_UCUI' if city=='unseen1' else 'meta_SCUI'
            for j in range(5):
                image=f'/tmp/{task}_{city}_{j}.png'
                label=float(i+1)+j*.1
                x=np.asarray([j/5.,i/5.,float(city=='unseen1')])
                sources.append({'task':task,'city':city,'image_path':image,
                                'x':x,'y':label})
                for expert in TASKS:
                    if expert==task:continue
                    old.append({'pseudo_task':task,'inner_partition':split,
                        'city':city,'tile_id':task+city+str(j),
                        'image_path':image,'expert':expert,
                        'knowledge_cache_key':task+city+str(j),
                        'knowledge_scores':[.5]*18,
                        'knowledge_masks':[1.]*18,
                        'label_log1p':label,'squared_error':100.,
                        'prediction_log1p':-10.,'error_log1p':-10.,
                        'semantic_source_mean':0.,'semantic_source_std':1.})
    return sources,old

class StableBankTests(unittest.TestCase):
    def test_per_task_training_and_fixed_hash(self):
        sources,_=evidence_fixture()
        bank=fit_bank(sources,TASKS,{'unseen1'},alpha=10.)
        self.assertEqual(len(bank),5)
        self.assertEqual({x['n_train'] for x in bank.values()},{10})
        self.assertEqual({x['n_common_unseen'] for x in bank.values()},{5})
        self.assertEqual(len({x['hash'] for x in bank.values()}),5)
        self.assertEqual(len(bank_report(bank)),5)
        # Change labels of task a ONLY: b/c/d/e expert coefficients and
        # own-task training normalization must remain EXACTLY frozen.
        altered=[dict(s) for s in sources]
        for s in altered:
            if s['task']=='a':s['y']+=3.
        changed=fit_bank(altered,TASKS,{'unseen1'},alpha=10.)
        for e in ('b','c','d','e'):
            np.testing.assert_array_equal(bank[e]['coef'],changed[e]['coef'])
            self.assertEqual(bank[e]['hash'],changed[e]['hash'])
            self.assertEqual(bank[e]['mu'],changed[e]['mu'])

    def test_pseudo_own_expert_excluded_and_common_scale(self):
        source,_=evidence_fixture()
        bank=fit_bank(source,TASKS,{'unseen1'})
        sample=source[0]
        out=predict_bank_row(sample,'a',TASKS,bank)
        self.assertEqual(set(out),set(TASKS)-{'a'})
        self.assertEqual(len({v['semantic_source_mean'] for v in out.values()}),1)
        self.assertEqual(len({v['semantic_source_std'] for v in out.values()}),1)
        for expert in out:
            self.assertEqual(out[expert]['expert_model_hash'],bank[expert]['hash'])

    def test_stable_evidence_paired_and_no_old_prediction_reuse(self):
        sources,old=evidence_fixture()
        bank=fit_bank(sources,TASKS,{'unseen1'})
        out=export_stable_evidence(old,sources,bank,TASKS,{'unseen1'})
        self.assertEqual(len(out),len(old))
        self.assertEqual(len(out),5*3*5*4)
        self.assertNotEqual(out[0]['prediction_log1p'],-10.)
        samples=aligned_samples(out,'outer',TASKS)
        self.assertEqual(len(samples),5*3*5)
        self.assertEqual(set(out[0]['knowledge_masks']),{1.})
        # Exactly same hash for named expert in ALL pseudo-task folds.
        for expert in TASKS:
            self.assertEqual({r['expert_model_hash'] for r in out
                              if r['expert']==expert},{bank[expert]['hash']})

    def test_refuses_inconsistent_city_split_and_tampered_label(self):
        source,old=evidence_fixture()
        bank=fit_bank(source,TASKS,{'unseen1'})
        wrong=[dict(r) for r in old]
        wrong[0]['inner_partition']='meta_UCUI'
        with self.assertRaises(ValueError):
            export_stable_evidence(wrong,source,bank,TASKS,{'unseen1'})
        wrong=[dict(r) for r in old]
        wrong[0]['label_log1p']+=5.
        with self.assertRaises(ValueError):
            export_stable_evidence(wrong,source,bank,TASKS,{'unseen1'})

    def test_persisted_bank_roundtrip_and_hash_includes_source_moments(self):
        import tempfile
        from pathlib import Path
        source,_=evidence_fixture()
        bank=fit_bank(source,TASKS,{'unseen1'})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bank.npz'
            save_bank(path,bank)
            loaded=load_bank(path)
            for task in TASKS:
                np.testing.assert_array_equal(
                    bank[task]['coef'],loaded[task]['coef'])
                self.assertEqual(bank[task]['hash'],loaded[task]['hash'])
            changed=[dict(x) for x in source]
            for row in changed:
                if row['task']=='a':row['y']+=3.
            retrained=fit_bank(changed,TASKS,{'unseen1'})
            # Centered expert weights can be invariant to label offset,
            # but the full source expert state (moments) must change.
            self.assertNotEqual(bank['a']['hash'],retrained['a']['hash'])

    def test_duplicate_training_or_missing_source_task_rejected(self):
        source,_=evidence_fixture()
        with self.assertRaises(ValueError):
            prepare_records(source+[dict(source[0])])
        with self.assertRaises(ValueError):
            fit_bank(source,TASKS+('outer',),{'unseen1'})
        with self.assertRaises(ValueError):
            fit_bank(source,TASKS,{'unseen1'},alpha=0.)

if __name__=='__main__':unittest.main()
