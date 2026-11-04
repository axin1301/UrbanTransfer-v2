import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from urban_transfer.ktrans_k7d import CUES
from urban_transfer.ktrans_empirical_competence import (
 identity,manifest_source_index,validate_inner_fold,
 validated_vlm_jobs,assemble_sample_rows,audit_identity)

def knowledge():
    return {'visual_cues':dict.fromkeys(CUES,2),
            'evidence_relevance':dict.fromkeys(CUES,3),
            'evidence_confidence':dict.fromkeys(CUES,2)}

class Bank:
    source_tasks=('a','b')
    task_to_all={'a':0,'b':1,'pseudo':2}
    routing=torch.tensor([[1.,0.],[0.,1.],[.4,.6]])
    source_means=torch.tensor([2.,4.])
    source_stds=torch.tensor([1.,2.])

class Data:
    def __init__(self):
        import pandas as pd
        self.df=pd.DataFrame({'indicator_column':['a','pseudo','b','pseudo'],
                              'city':['old','old','new','new']})
        self.base=SimpleNamespace(paths=['/tmp/a.png','/tmp/s1.png','/tmp/b.png','/tmp/s2.png'])

class TestEmpiricalCompetence(unittest.TestCase):
    def setUp(self):
        self.ds=Data()
        self.splits={'train':[0],'meta_SCUI':[1],'meta_UCSI':[2],'meta_UCUI':[3]}
        self.allowed={identity(self.ds.base.paths[i],self.ds.df.iloc[i].indicator_column)
                      for i in range(4)}
        self.meta={'heldout_task':'pseudo','train_csv':'/tmp/inner_source_only.csv',
                   'normalized_experts':True,'expert_bank':True}
        self.rows=[{'task':'pseudo','tile_id':'one','index':10,'city':'old',
                    'image_path':'/tmp/s1.png'},
                   {'task':'pseudo','tile_id':'two','index':20,'city':'new',
                    'image_path':'/tmp/s2.png'}]
        self.vlm={('one','pseudo','image_task'):{'key':'hash1','parsed':knowledge()},
                  ('two','pseudo','image_task'):{'key':'hash2','parsed':knowledge()}}

    def test_source_only_inner_fold_partition(self):
        partitions=validate_inner_fold('pseudo',self.meta,self.ds,self.splits,
            Bank(),self.allowed,['a','b','pseudo'],'outer','/tmp/inner_source_only.csv')
        self.assertEqual(len(partitions['meta_UCUI']),1)
        self.assertEqual(len(partitions['meta_SCUI']),1)

    def test_reject_wrong_inner_dataset(self):
        bad={**self.meta,'train_csv':'/tmp/unrelated.csv'}
        with self.assertRaises(ValueError):
            validate_inner_fold('pseudo',bad,self.ds,self.splits,Bank(),self.allowed,
                                ['a','b','pseudo'],'outer','/tmp/inner_source_only.csv')

    def test_reject_pseudo_target_in_source_expert(self):
        class WrongBank(Bank):
            source_tasks=('a','pseudo')
        with self.assertRaises(ValueError):
            validate_inner_fold('pseudo',self.meta,self.ds,self.splits,
             WrongBank(),self.allowed,['a','b','pseudo'],'outer','/tmp/inner_source_only.csv')

    def test_expert_error_and_source_only_semantic_moments(self):
        z=np.array([[.1,.3],[.2,.4]])
        y=np.array([3.2,3.6])
        rows,counts=assemble_sample_rows('pseudo','meta_SCUI',self.rows,
                                          z,y,Bank(),self.vlm)
        self.assertEqual(counts['with_vlm'],2)
        self.assertEqual(len(rows),4)
        self.assertEqual(audit_identity(rows,'outer'),4)
        r=rows[0]
        self.assertAlmostEqual(r['semantic_source_mean'],3.2,places=5)
        self.assertAlmostEqual(r['semantic_source_std'],1.6,places=5)
        self.assertAlmostEqual(r['prediction_log1p'],3.36,places=5)
        self.assertAlmostEqual(r['squared_error'],(3.36-3.2)**2,places=5)
        self.assertEqual(len(r['knowledge_scores']),18)
        self.assertEqual(len(r['knowledge_masks']),18)

    def test_missing_vlm_excluded_with_coverage(self):
        rows,cov=assemble_sample_rows('pseudo','meta_UCUI',self.rows,
               np.ones((2,2)),np.ones(2),Bank(),{'one':{}})
        self.assertEqual(cov.get('missing_image_task_vlm'),2)
        self.assertEqual(len(rows),0)

    def test_outer_manifest_train_only_and_duplicate_rejection(self):
        m={'heldout_task':'outer','rows':[
           {'partition':'train','task':'pseudo','image_path':'/tmp/s1.png','index':10},
           {'partition':'meta_UCUI','task':'outer','image_path':'/tmp/target.png','index':30}]}
        self.assertEqual(len(manifest_source_index(m)),1)
        m['rows'].append(m['rows'][0])
        with self.assertRaises(ValueError):manifest_source_index(m)

    def test_reject_outer_task_in_data(self):
        rows,_=assemble_sample_rows('pseudo','meta_SCUI',self.rows,
              np.ones((2,2)),np.ones(2),Bank(),self.vlm)
        with self.assertRaises(ValueError):
            audit_identity([{**rows[0],'pseudo_task':'outer'}],'outer')
if __name__=='__main__':unittest.main()
