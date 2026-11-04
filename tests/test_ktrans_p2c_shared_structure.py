import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from urban_transfer.ktrans_p2a import samples_from_rows
from urban_transfer.ktrans_p2c1 import (
    standardize_effects,task_pair_structure,rank_structure,task_semantic_association)
from urban_transfer.ktrans_p2c2 import (
    validated_fold,tensors_for_expert,fold_parameter_audit,
    training_data_audit)

TASKS=('a','b','c','d','e')
CUES=6

def records():
    rows=[]
    for task in TASKS:
        experts=[e for e in TASKS if e!=task]
        for split in ('meta_SCUI','meta_UCUI'):
            for city in ('seen1','seen2') if split=='meta_SCUI' else ('new1','new2'):
                for i in range(10):
                    high=i>=5
                    scores=[.5]*18
                    scores[0]=1. if high else 0.
                    # Other five cues are all tied at median and thus
                    # correctly excluded from the complete-case tensor.
                    for j,expert in enumerate(experts):
                        loss=float(j+1+(.5 if high else -.5)*(j+1))
                        rows.append({'pseudo_task':task,'inner_partition':split,
                            'tile_id':f'{task}:{split}:{city}:{i}',
                            'city':city,'expert':expert,'squared_error':loss,
                            'label_log1p':2.,'image_path':f'/tmp/{task}{split}{city}{i}.png',
                            'knowledge_scores':scores,'knowledge_masks':[1.]*18,
                            'knowledge_cache_key':f'key-{task}-{split}-{city}-{i}'})
    return rows

def checkpoints(csv_path):
    result={}
    for i,task in enumerate(TASKS):
        src=[name for name in TASKS if name!=task]
        state={'shared.1.weight':torch.tensor([1.+i,2.])}
        for k,name in enumerate(src):
            state[f'experts.{k}.0.weight']=torch.tensor([float(k+i),2.])
            state[f'experts.{k}.0.bias']=torch.tensor([float(k)])
        result[task]={'epoch':30,'meta':{'heldout_task':task,
            'source_tasks':src,'train_csv':str(csv_path),
            'seed':42,'heldout_cities':['new1'],
            'source_stats':{name:(1.,1.) for name in src},
            'expert_bank':True,'normalized_experts':True},
            'model':state}
    return result

class P2CSharingTests(unittest.TestCase):
    def test_conditional_tensor_missing_diagonal_and_masks(self):
        samples=samples_from_rows(records(),'outer')
        names,tensor,mask,detail,thresholds,scales=standardize_effects(
            samples,TASKS,min_group=5)
        self.assertEqual(len(samples),200)
        self.assertEqual(tensor['meta_SCUI'].shape,(5,5,6))
        for i in range(5):
            self.assertFalse(mask['meta_SCUI'][i,i].any())
            self.assertTrue(np.isnan(tensor['meta_SCUI'][i,i]).all())
        self.assertTrue(np.isfinite(tensor['meta_SCUI'][0,1,0]))
        self.assertFalse(mask['meta_SCUI'][0,1,1])
        self.assertEqual(len(detail),40)
        self.assertEqual(thresholds[('a','built_up')],.5)
        self.assertTrue(all(v>0 for v in scales.values()))

    def test_task_pair_uses_only_common_experts_and_observed_cues(self):
        names,tensor,mask,_,_,_=standardize_effects(
            samples_from_rows(records(),'outer'),TASKS,min_group=5)
        texts={t:[.2,.3,.4,.5,.6,.7] for t in TASKS}
        similarity=task_pair_structure(names,tensor,mask,texts)
        self.assertEqual(len(similarity),20)
        self.assertTrue(all(r['n_common_named_experts']==3 for r in similarity))
        self.assertTrue(all(r['n_effect_entries_shared']==3 for r in similarity))
        low=rank_structure(names,tensor,mask)
        self.assertEqual(len(low),10)
        self.assertTrue(all(r['n_common_named_experts']==2 for r in low))
        self.assertTrue(all(r['n_common_entries']==2 for r in low))
        self.assertEqual(task_semantic_association(similarity[:10])['n_pairs'],10)

    def test_inner_model_provenance_and_weight_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'source.csv'
            c=checkpoints(path)
            row=validated_fold(c['a'],'a',TASKS,'outer',path)
            self.assertEqual(row['source_expert_count'],4)
            folds,pairs,shared=fold_parameter_audit(c,TASKS,'outer',path)
            self.assertEqual(len(folds),5)
            self.assertEqual(len(pairs),30)
            self.assertEqual(len(shared),10)
            # A named expert can move between fold-local expert indices:
            # e.g., expert "c" is index 1 in fold "a", index 1 in
            # fold "b"; expert "b" is unavailable in fold "b".
            # Compare both sides explicitly using the SAME name.
            from urban_transfer.ktrans_p2c2 import compare_vectors
            actual=next(row for row in pairs if
                        row['pseudo_a']=='a' and row['pseudo_b']=='b'
                        and row['named_expert']=='c')
            expected=compare_vectors(tensors_for_expert(c['a'],'c'),
                                     tensors_for_expert(c['b'],'c'))
            self.assertEqual(actual['n_parameters'],expected['n_parameters'])
            self.assertAlmostEqual(actual['relative_l2_parameter_shift'],
                                   expected['relative_l2_parameter_shift'])
            # The named expert is indexed by its local fold-specific
            # position, not by a supposedly global expert.0 head.
            p=tensors_for_expert(c['a'],'b')
            self.assertIn('0.weight',p)
            self.assertTrue(any(not r['identical_parameters'] for r in pairs))
            bad=checkpoints(path)
            bad['a']['meta']['source_tasks'].append('outer')
            with self.assertRaises(ValueError):
                fold_parameter_audit(bad,TASKS,'outer',path)

    def test_supervised_training_overlap_no_heldout_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.csv'
            data=[]
            for t in TASKS:
                for city in ('seen1','seen2','new1'):
                    data.append({'indicator_column':t,'city':city,
                                 'image_name':t+city+'.png'})
            pd.DataFrame(data).to_csv(path,index=False)
            folds,_,_=fold_parameter_audit(checkpoints(path),TASKS,'outer',path)
            counts,overlap=training_data_audit(folds,path,TASKS,'outer')
            self.assertEqual(len(counts),20)
            self.assertEqual(len(overlap),10)
            self.assertTrue(all(r['n_supervised_source_rows']==2 for r in counts))
            data.append({'indicator_column':'outer','city':'seen1',
                         'image_name':'outer.png'})
            pd.DataFrame(data).to_csv(path,index=False)
            with self.assertRaises(ValueError):
                training_data_audit(folds,path,TASKS,'outer')

if __name__=='__main__':unittest.main()
