"""P2-E1: frozen indicator-relational VLM prompt, jobs, validation and pilot summary.

Three escalating, paired prompt arms, plus shuffled relationship negative
control. TEXT-only prompts cannot see images or task labels. Image prompts
use source-only pseudo-task city/tile evidence. Expert competence is a
hypothesis; the VLM is never told observed expert errors.
"""
import hashlib,json
from collections import defaultdict
import numpy as np
from urban_transfer.ktrans_p2e1_indicator_spec import expert_cards,valid_spec_for_pilot

VERSION='ktrans-p2e1-indicator-relation-v1'
ARMS=('definition_only','definition_relation','definition_relation_region',
      'permuted_relation_region')
SYSTEM=('You are a conservative urban indicator methodology analyst. '
        'Your task is to propose HYPOTHESES about transfer from fixed source-task predictors. '
        'No ground truth indicator value, expert prediction, expert error, fitted weights '
        'or actual expert accuracy is available to you. Do not invent these. '
        'Satellite imagery can show morphology but does not identify actual schools, '
        'cell towers, population or measured accessibility. '
        'Unknown units/formulas are explicitly unverified; never make them up. '
        'Return a single strict JSON object only, without markdown.')
RESPONSE=(
 'For EACH of the four candidate source experts, return a JSON object under "experts" '
 'keyed by the exact expert ID, with keys "compatibility" (integer 0..3 or null), '
 '"region_support" (integer 0..3 or null; MUST be null if no image), '
 '"transfer_risk" (integer 0..3 or null), '
 '"complementarity" (max 100 character text), '
 '"uncertainty" (integer 0..3), and "reason" (max 180 character text). '
 'Provide "limitations" (max 240 characters) at top level. '
 'If relation data are hypotheses, explicitly say that empirical transfer '
 'is unverified; do not report numeric accuracy, target values, or fake units.')

def sample_source_rows(evidence_rows,source_tasks,per_task_split=4,seed=42):
    """Deterministic hash-stratified pilot, no label/loss based selection."""
    if per_task_split<1:raise ValueError('Need positive pilot tiles per task/split')
    groups=defaultdict(dict)
    names=set(source_tasks)
    for row in evidence_rows:
        task=row['pseudo_task'];split=row['inner_partition']
        if task not in names or split not in ('meta_SCUI','meta_UCUI'):
            raise ValueError('Non-source pilot evidence')
        key=(row['tile_id'],row['image_path'],row['city'])
        slot=groups[task,split]
        if row['tile_id'] in slot and slot[row['tile_id']]!=key:
            raise ValueError('Duplicate tile identity')
        slot[row['tile_id']]=key
    result=[]
    for task in sorted(names):
        for split in ('meta_SCUI','meta_UCUI'):
            pool=list(groups.get((task,split),{}).items())
            if len(pool)<per_task_split:raise ValueError('Insufficient pilot sample '+task+split)
            pool.sort(key=lambda p:hashlib.sha256(
                f'{seed}|{task}|{split}|{p[0]}'.encode()).hexdigest())
            for tile,(tile_id,path,city) in pool[:per_task_split]:
                result.append({'task':task,'split':split,'tile_id':tile_id,
                               'image_path':path,'city':city})
    return result

def relation_permutation(task,sources,spec):
    """A deterministic wrong-relation control: rotate relation cards ONLY.

    Keep expert identity, definition, units and target unchanged.
    Rotation avoids presenting a true expert's own computed edge.
    """
    cards=expert_cards(task,sources,spec,True)
    relations=[card['relation_to_target'] for card in cards]
    if len(cards)<2:raise ValueError('Need multiple experts')
    for j,card in enumerate(cards):
        card['relation_to_target']={**relations[(j+1)%len(cards)],
             'control_note':'Shuffled across candidate experts: INVALID relation control'}
    return cards

def prompt(task,source_tasks,spec,arm):
    if arm not in ARMS:raise ValueError('Invalid P2-E pilot arm')
    target=valid_spec_for_pilot(task,spec)
    relation=arm!='definition_only'
    cards=(relation_permutation(task,source_tasks,spec) if arm=='permuted_relation_region'
           else expert_cards(task,source_tasks,spec,relation))
    context={'target_indicator':{k:target[k] for k in
             ('indicator','definition','measurement_context','canonical_description',
              'raw_unit','exact_formula','raw_statistical_basis','label_transform','measurement_source')},
             'source_experts':cards,
             'task_relation_role':'Relation edges are conceptual HYPOTHESES, not measured accuracy.' if relation else
                                  'No indicator relationship hypotheses supplied.'}
    has_image=arm.endswith('_region')
    return ('Experimental arm: '+arm+'\n'
       +'Image available: '+str(has_image)+'. '
       +('Analyze ONLY visible morphology and adjust source expert transfer hypotheses to this specific region. '
         if has_image else 'NO IMAGE is provided. Evaluate only global target/source measurement compatibility. '
         'Set region_support=null for EVERY expert. ')
       +'Source registry facts (NOT measured target labels):\n'
       +json.dumps(context,ensure_ascii=False,sort_keys=True)
       +'\n'+RESPONSE)

def make_jobs(sampled,source_tasks,spec,model,max_pixels):
    tasks=tuple(sorted(source_tasks))
    jobs=[]
    for task in tasks:
        for arm in ARMS[:2]:
            text=prompt(task,tasks,spec,arm)
            identity={'version':VERSION,'arm':arm,'task':task,'tile_id':None,
                      'model':model,'max_pixels':max_pixels,'prompt':text}
            jobs.append({**identity,'key':hashlib.sha256(
                json.dumps(identity,sort_keys=True).encode()).hexdigest(),
                'image_path':None,'split':None,'city':None,'prompt':text})
    for sample in sampled:
        for arm in ARMS[2:]:
            task=sample['task'];text=prompt(task,tasks,spec,arm)
            identity={'version':VERSION,'arm':arm,'task':task,
                      'tile_id':sample['tile_id'],'model':model,
                      'max_pixels':max_pixels,'prompt':text}
            jobs.append({**identity,'key':hashlib.sha256(
                json.dumps(identity,sort_keys=True).encode()).hexdigest(),
                'image_path':sample['image_path'],'split':sample['split'],
                'city':sample['city'],'prompt':text})
    if len({j['key'] for j in jobs})!=len(jobs):
        raise ValueError('Duplicate VLM prompt keys')
    return jobs

def parse(raw,source_tasks,task,has_image):
    try:obj=json.loads(raw)
    except (json.JSONDecodeError,TypeError):return None
    if not isinstance(obj,dict) or not isinstance(obj.get('experts'),dict):
        return None
    expected=set(source_tasks)-{task}
    if set(obj['experts'])!=expected or len(expected)!=4:return None
    normalized={}
    for expert in sorted(expected):
        row=obj['experts'][expert]
        if not isinstance(row,dict):return None
        expected_fields={'compatibility','region_support','transfer_risk',
                         'complementarity','uncertainty','reason'}
        if set(row)!=expected_fields:return None
        for field in ('compatibility','region_support','transfer_risk','uncertainty'):
            v=row[field]
            if v is not None and (type(v) is not int or v not in (0,1,2,3)):
                return None
        if not has_image and row['region_support'] is not None:return None
        if row['uncertainty'] is None:return None
        if not isinstance(row['reason'],str) or len(row['reason'])>180:return None
        if not isinstance(row['complementarity'],str) or len(row['complementarity'])>100:return None
        normalized[expert]=row
    if not isinstance(obj.get('limitations'),str) or len(obj['limitations'])>240:
        return None
    return {'experts':normalized,'limitations':obj['limitations']}

def descriptive_audit(records,source_tasks):
    """VLM descriptive outputs only. DO NOT score on heldout UCUI labels."""
    rows=[]
    for row in records:
        parsed=row.get('parsed')
        if parsed is None:continue
        for expert,profile in parsed['experts'].items():
            rows.append({'arm':row['arm'],'task':row['task'],'split':row.get('split'),
                'tile_id':row.get('tile_id'),'city':row.get('city'),
                'expert':expert,
                'compatibility':profile['compatibility'],
                'region_support':profile['region_support'],
                'transfer_risk':profile['transfer_risk'],
                'uncertainty':profile['uncertainty']})
    return rows
