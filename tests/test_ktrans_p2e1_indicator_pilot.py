import json,tempfile,unittest
from pathlib import Path
from urban_transfer.ktrans_p2e1_indicator_spec import catalog,relationship,expert_cards
from urban_transfer.ktrans_p2e1_pilot import (
    ARMS,sample_source_rows,make_jobs,parse,prompt,relation_permutation)

TASKS=('population_count','nighttime_light_intensity',
       'food_accessibility','digital_modernization_tmi',
       'digital_total_cell_intensity')

def rows():
    result=[]
    for task in TASKS:
        for split,city in (('meta_SCUI','seen'),('meta_UCUI','new')):
            for i in range(6):
                for expert in TASKS:
                    if expert!=task:
                        result.append({'pseudo_task':task,'inner_partition':split,
                            'tile_id':f'{task}-{split}-{i}',
                            'image_path':f'/tmp/{task}_{split}_{i}.png',
                            'city':city,'expert':expert,
                            'label_log1p':float(i),
                            'squared_error':float(i+1)})
    return result

def fake(task,has_image=True):
    return {'experts':{t:{'compatibility':2,'region_support':1 if has_image else None,
                            'transfer_risk':1,'complementarity':'Maybe distinct morphology',
                            'uncertainty':2,'reason':'Conceptual hypothesis only'}
                       for t in TASKS if t!=task},'limitations':'Not measured expert accuracy'}

class IndicatorPilotTests(unittest.TestCase):
    def test_unknown_units_never_fabricated_and_relations_hypotheses(self):
        spec=catalog()
        self.assertEqual(len(spec),10)
        self.assertTrue(all(s['raw_unit']=='unverified' and
                            s['exact_formula']=='unverified' for s in spec.values()))
        self.assertEqual(relationship('education_accessibility','food_accessibility',spec)['type'],
                         'shared_accessibility_formulation')
        self.assertFalse(relationship('education_accessibility','food_accessibility',spec)['verified_empirical_transfer'])
        self.assertEqual(len(expert_cards('population_count',TASKS,spec,True)),4)

    def test_reviewed_methodology_requires_all_ten_and_source(self):
        spec=catalog()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'units.json'
            p.write_text(json.dumps({'population_count':{'raw_unit':'people'}}))
            with self.assertRaises(ValueError):catalog(p)
            full={k:{'raw_unit':'unverified','exact_formula':'unverified',
                     'raw_statistical_basis':'unverified','source_reference':'user verified data dictionary'}
                  for k in spec}
            p.write_text(json.dumps(full))
            revised=catalog(p)
            self.assertEqual(set(revised),set(spec))

    def test_hash_selection_does_not_depend_on_errors_or_labels(self):
        source=rows()
        a=sample_source_rows(source,TASKS,per_task_split=3,seed=42)
        b=sample_source_rows([{**r,'squared_error':-500.,'label_log1p':999.}
                              for r in reversed(source)],TASKS,3,42)
        self.assertEqual(a,b)
        self.assertEqual(len(a),30)
        self.assertEqual(sum(s['split']=='meta_UCUI' for s in a),15)

    def test_nested_prompt_conditions_image_vs_no_image(self):
        spec=catalog()
        jobs=make_jobs(sample_source_rows(rows(),TASKS,2),TASKS,spec,'fake-vlm',262144)
        self.assertEqual(len(jobs),10+20*2)
        self.assertEqual(len({j['key'] for j in jobs}),len(jobs))
        for job in jobs:
            has_image=job['arm'].endswith('_region')
            self.assertEqual(job['image_path'] is not None,has_image)
            self.assertIn('unverified',job['prompt'])
            self.assertNotIn('squared_error',job['prompt'])
            self.assertNotIn('label_log1p',job['prompt'])
        self.assertNotIn('relation_to_target',prompt(TASKS[0],TASKS,spec,'definition_only'))
        self.assertIn('relation_to_target',prompt(TASKS[0],TASKS,spec,'definition_relation'))
        perm=relation_permutation(TASKS[0],TASKS,spec)
        self.assertTrue(all('control_note' not in row['relation_to_target'] for row in perm))
        self.assertNotIn('permuted_relation_region',prompt(TASKS[0],TASKS,spec,'permuted_relation_region'))

    def test_manifest_roundtrip_accepts_existing_dry_run_and_rejects_changes(self):
        from scripts.run_ktrans_p2e1_indicator_pilot import freeze_manifest
        spec=catalog()
        jobs=make_jobs(sample_source_rows(rows(),TASKS,1),TASKS,spec,
                       'fake-vlm',262144)
        manifest={'plan':{'source_tasks':tuple(TASKS),'seed':42},
                  'jobs':jobs,'indicator_specs':spec}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'pilot_manifest.json'
            freeze_manifest(path,manifest)
            old_bytes=path.read_bytes()
            # Simulates CPU --dry-run followed by GPU --max-new-jobs 8
            # with identical arguments; tuples become JSON lists.
            freeze_manifest(path,manifest)
            self.assertEqual(path.read_bytes(),old_bytes)
            modified={'plan':{'source_tasks':tuple(TASKS),'seed':43},
                      'jobs':jobs,'indicator_specs':spec}
            with self.assertRaisesRegex(ValueError,'manifest differs'):
                freeze_manifest(path,modified)
            self.assertEqual(path.read_bytes(),old_bytes)

    def test_qwen_fenced_json_parser_and_invalid_schema_rejection(self):
        from urban_transfer.ktrans_p2e1_pilot import parse_json_object
        task=TASKS[0]
        answer=json.dumps(fake(task,False))
        self.assertIsNotNone(parse('```json\\n'+answer+'\\n```',
                                   TASKS,task,False))
        self.assertIsNotNone(parse('Generated JSON:\\n'+answer,
                                   TASKS,task,False))
        self.assertIsNone(parse(answer+'\\n'+answer,TASKS,task,False))
        self.assertIsNone(parse('{"experts":',TASKS,task,False))
        self.assertIsNone(parse_json_object(None))
        malformed=fake(task,False)
        malformed['experts'][TASKS[1]]['region_support']=2
        self.assertIsNone(parse(json.dumps(malformed),TASKS,task,False))

    def test_batch_grouping_and_per_job_identity(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.run_ktrans_p2e1_indicator_pilot import run_batched_jobs
        spec=catalog()
        jobs=make_jobs(sample_source_rows(rows(),TASKS,1),TASKS,spec,
                       'Qwen/Qwen3-VL-32B-Instruct',262144)
        # 10 text-only + 10? actually five source tasks * two image
        # arms * two splits = 20 image requests.
        outputs=[]
        calls=[]
        a=SimpleNamespace(batch_size=8,max_new_tokens=1536,source_tasks=TASKS)
        def mock_infer(model,processor,chunk,max_new):
            calls.append(list(chunk))
            return [json.dumps(fake(j['task'],j['image_path'] is not None))
                    for j in chunk]
        with patch('urban_transfer.ktrans_p2e1_pilot.parse', wraps=parse):
            with patch('scripts.run_ktrans_p2e1_indicator_pilot.infer_batch',
                       side_effect=mock_infer):
                run_batched_jobs(None,None,jobs,a,outputs.append)
        self.assertEqual(len(outputs),len(jobs))
        self.assertEqual([x['key'] for x in outputs],
                         [x['key'] for x in jobs])
        self.assertTrue(all(x['parsed'] is not None for x in outputs))
        self.assertTrue(all(len(chunk)<=8 for chunk in calls))
        self.assertTrue(all(len({j['image_path'] is not None for j in chunk})==1
                            for chunk in calls))
        self.assertTrue(all(x['model']=='Qwen/Qwen3-VL-32B-Instruct'
                            for x in outputs))

    def test_strict_json_parse_and_unknown_capability(self):
        task=TASKS[0]
        self.assertIsNotNone(parse(json.dumps(fake(task,False)),TASKS,task,False))
        self.assertIsNone(parse(json.dumps(fake(task,True)),TASKS,task,False))
        self.assertIsNotNone(parse(json.dumps(fake(task,True)),TASKS,task,True))
        wrong=fake(task,True);wrong['experts'][TASKS[1]]['compatibility']=4
        self.assertIsNone(parse(json.dumps(wrong),TASKS,task,True))
        wrong=fake(task,True);wrong['experts'][task]=wrong['experts'][TASKS[1]]
        self.assertIsNone(parse(json.dumps(wrong),TASKS,task,True))

if __name__=='__main__':unittest.main()
