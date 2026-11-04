import unittest
from pathlib import Path
from types import SimpleNamespace
import hashlib
import pandas as pd
from urban_transfer.ktrans_source_coverage import (
    full_source_manifest,validate_full_source_manifest,coverage_plan)
from scripts.cache_ktrans_k7d import unique_jobs,get_pending

class FullSourceCoverageTests(unittest.TestCase):
    def setUp(self):
        self.ds=SimpleNamespace(
           df=pd.DataFrame({'indicator_column':['src_a','src_b','target','src_a'],
                            'city':['known','known','new','known']}),
           base=SimpleNamespace(paths=['/tmp/img_a.png','/tmp/img_b.png',
                    '/tmp/target.png','/tmp/img_c.png']))
        self.splits={'train':[0,1,3],'meta_SCUI':[],
                     'meta_UCSI':[],'meta_UCUI':[2]}
        self.meta={'source_tasks':['src_a','src_b'],'heldout_task':'target',
                   'heldout_cities':['new'],'seed':42}
        self.path=Path('/tmp/k4/best.pt')

    def test_only_outer_train_exported_and_complete(self):
        m=full_source_manifest(self.ds,self.splits,self.meta,self.path)
        self.assertEqual(len(m['rows']),3)
        self.assertEqual({r['task'] for r in m['rows']},{'src_a','src_b'})
        self.assertTrue(all(r['partition']=='train' for r in m['rows']))
        self.assertEqual(validate_full_source_manifest(m,self.ds,self.splits,
                        self.meta,self.path),3)
        self.assertNotIn('/tmp/target.png',[r['image_path'] for r in m['rows']])

    def test_job_dedup_by_unique_image_and_task(self):
        # Identical tile across multiple task rows: one image_only, two image_task.
        self.ds.base.paths[1]='/tmp/img_a.png'
        m=full_source_manifest(self.ds,self.splits,self.meta,self.path)
        jobs=unique_jobs(m,'Qwen/Qwen3-VL-8B-Instruct',262144)
        kinds=[j['condition'] for j in jobs]
        self.assertEqual(kinds.count('image_only'),2)
        self.assertEqual(kinds.count('image_task'),3)
        self.assertEqual(kinds.count('text_only'),2)

    def test_pending_and_exact_record_provenance(self):
        m=full_source_manifest(self.ds,self.splits,self.meta,self.path)
        jobs=unique_jobs(m,'Qwen/Qwen3-VL-8B-Instruct',262144)
        first=jobs[0]
        bad={first['key']:{**first,'parsed':{'ok':True},'prompt':'wrong'}}
        self.assertEqual(len(get_pending(jobs,bad)),len(jobs))
        plan,pending=coverage_plan(m,bad,'Qwen/Qwen3-VL-8B-Instruct',262144)
        self.assertEqual(sum(plan['jobs_cached_valid'].values()),0)
        self.assertEqual(len(pending),len(jobs))
        good={first['key']:{**first,'parsed':{'ok':True}}}
        plan,pending=coverage_plan(m,good,'Qwen/Qwen3-VL-8B-Instruct',262144)
        self.assertEqual(sum(plan['jobs_cached_valid'].values()),1)
        self.assertEqual(len(pending),len(jobs)-1)

    def test_manifest_tampering_and_heldout_fail_closed(self):
        m=full_source_manifest(self.ds,self.splits,self.meta,self.path)
        m['rows'].pop()
        with self.assertRaises(ValueError):
            validate_full_source_manifest(m,self.ds,self.splits,self.meta,self.path)
        self.ds.df.loc[0,'indicator_column']='target'
        with self.assertRaises(ValueError):
            full_source_manifest(self.ds,self.splits,self.meta,self.path)

    def test_no_heldout_cities_in_outer_train(self):
        self.ds.df.loc[0,'city']='new'
        with self.assertRaises(ValueError):
            full_source_manifest(self.ds,self.splits,self.meta,self.path)

if __name__=='__main__':unittest.main()
