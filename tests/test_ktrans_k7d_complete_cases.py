import unittest
from scripts.prepare_ktrans_k7d_complete_cases import complete_cases
from scripts.cache_ktrans_k7d import unique_jobs

class CompleteCasesTests(unittest.TestCase):
    def setUp(self):
        rows=[]
        for i in range(4):
            rows.append({'index':i,'partition':'train','tile_id':f'train{i}',
                         'image_path':f'/tmp/train{i}.png','city':'seen',
                         'task':'education_accessibility'})
        rows.extend([
            {'index':5,'partition':'meta_SCUI','tile_id':'bad',
             'image_path':'/tmp/bad.png','city':'seen','task':'healthcare_accessibility'},
            {'index':6,'partition':'meta_SCUI','tile_id':'good',
             'image_path':'/tmp/good.png','city':'seen','task':'healthcare_accessibility'},
            {'index':7,'partition':'meta_UCSI','tile_id':'ucsi',
             'image_path':'/tmp/ucsi.png','city':'unseen','task':'education_accessibility'},
            {'index':8,'partition':'meta_UCUI','tile_id':'bad',
             'image_path':'/tmp/bad.png','city':'unseen','task':'healthcare_accessibility'},
            {'index':9,'partition':'meta_UCUI','tile_id':'ucui',
             'image_path':'/tmp/ucui.png','city':'unseen','task':'healthcare_accessibility'},
        ])
        # A tile should never denote different images, regardless of partition.
        self.manifest={'version':'k7d-manifest-v1',
                       'source_tasks':['education_accessibility'],
                       'heldout_task':'healthcare_accessibility',
                       'partition_counts':{'train':4,'meta_SCUI':2,
                                           'meta_UCSI':1,'meta_UCUI':2},
                       'rows':rows}
        self.jobs=unique_jobs(self.manifest,'model',262144)
        self.records={j['key']:{**j,'parsed':{'visual_cues':{}}} for j in self.jobs}

    def test_rejects_incomplete_tile_from_all_partitions(self):
        for job in self.jobs:
            if job['tile_id']=='bad' and job['condition']=='image_only':
                self.records[job['key']]['parsed']=None
        selected,excluded=complete_cases(self.manifest,self.records,'model',262144)
        self.assertEqual(len(excluded),2)
        self.assertEqual(selected['partition_counts']['train'],4)
        self.assertEqual(selected['partition_counts']['meta_SCUI'],1)
        self.assertEqual(selected['partition_counts']['meta_UCUI'],1)
        self.assertTrue(all(r['tile_id']!='bad' for r in selected['rows']))
        self.assertEqual(self.manifest['partition_counts']['meta_SCUI'],2)
        self.assertEqual(selected['complete_case_protocol']['removed_rows'],2)
        self.assertEqual(selected['complete_case_protocol']['missing_conditions_row_counts']['image_only'],2)

    def test_fails_when_eval_partition_emptied(self):
        for job in self.jobs:
            if job['tile_id']=='ucsi' and job['condition']=='image_only':
                self.records[job['key']]['parsed']=None
        with self.assertRaisesRegex(ValueError,'empty a partition'):
            complete_cases(self.manifest,self.records,'model',262144)

    def test_fails_when_source_train_task_eliminated(self):
        for job in self.jobs:
            if job['condition']=='text_only' and job['task']=='education_accessibility':
                self.records[job['key']]['parsed']=None
        with self.assertRaises(ValueError):
            complete_cases(self.manifest,self.records,'model',262144)

if __name__=='__main__':unittest.main()
