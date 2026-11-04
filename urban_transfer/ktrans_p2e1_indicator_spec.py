"""P2-E1 indicator specification registry: source-verified descriptions, NEVER invented units.

Task definition and conceptual group derive from protocol.py/task_profiles.py.
Every physical raw unit and exact formula is UNKNOWN until a separately
reviewed dataset-methodology JSON specifies them. Dataset's supervised
log1p representation is known, but log1p does NOT reveal the original units.
"""
import json
from pathlib import Path
from urban_transfer.protocol import TASK_DESCRIPTIONS
from urban_transfer.task_profiles import PROFILES

FAMILIES={
 'population_count':'population',
 'nighttime_light_intensity':'luminosity',
 'education_accessibility':'accessibility',
 'food_accessibility':'accessibility',
 'healthcare_accessibility':'accessibility',
 'mobility_opportunity':'accessibility',
 'social_opportunity':'accessibility',
 'digital_modernization_tmi':'telecom',
 'digital_total_cell_intensity':'telecom',
 'digital_lte_intensity':'telecom'}
UNKNOWN='unverified'

def catalog(reviewed=None):
    """Only reviewed user-supplied unit/formula may replace unknown fields."""
    if set(TASK_DESCRIPTIONS)!=set(FAMILIES) or set(PROFILES)!=set(FAMILIES):
        raise ValueError('Source task definitions out of sync')
    result={}
    for task in sorted(FAMILIES):
        definition,measurement,drivers=PROFILES[task]
        result[task]={'indicator':task,'definition':definition,
            'measurement_context':measurement,'spatial_hypotheses':drivers,
            'canonical_description':TASK_DESCRIPTIONS[task],
            'family':FAMILIES[task],
            'raw_unit':UNKNOWN,'exact_formula':UNKNOWN,
            'raw_statistical_basis':UNKNOWN,
            'label_transform':'log1p(raw nonnegative indicator)',
            'measurement_source':'repository protocol.py and task_profiles.py; raw unit and formula UNVERIFIED'}
    if reviewed is not None:
        spec=json.loads(Path(reviewed).read_text())
        if not isinstance(spec,dict) or set(spec)!=set(result):
            raise ValueError('Reviewed methodology must cover exactly all ten indicators')
        for task,entry in spec.items():
            if not isinstance(entry,dict) or set(entry)!={'raw_unit','exact_formula','raw_statistical_basis','source_reference'}:
                raise ValueError('Explicit reviewed unit, formula, basis and reference required for '+task)
            if any(not isinstance(v,str) or not v.strip() for v in entry.values()):
                raise ValueError('Empty reviewed methodology field for '+task)
            if entry['source_reference'].strip().lower() in ('unknown','unverified','n/a'):
                raise ValueError('Missing review source '+task)
            result[task].update(entry)
            result[task]['measurement_source']='reviewed methodology: '+entry['source_reference']
    return result

# Relationship edges are HYPOTHESES derived from indicator descriptions,
# not causal facts, learned source errors or measured correlations.
def relationship(target,source,spec):
    if target==source:raise ValueError('Own-task expert must be excluded')
    if target not in spec or source not in spec:raise ValueError('Unknown indicator')
    a=spec[target]['family'];b=spec[source]['family']
    if a=='accessibility' and b=='accessibility':
        return {'type':'shared_accessibility_formulation',
            'hypothesis':'Both describe population-normalized distance-decay facility accessibility; facility categories differ.',
            'important_difference':'Facility type, distribution and exact facility taxonomy may differ.',
            'verified_empirical_transfer':False}
    if a=='telecom' and b=='telecom':
        return {'type':'related_telecom_infrastructure',
            'hypothesis':'Both concern cellular infrastructure, but technology generations/modernization and all-cell intensity need not agree.',
            'important_difference':'TMI definition, normalization and LTE/total construction must be verified before asserting subset identities.',
            'verified_empirical_transfer':False}
    return {'type':'possible_shared_urban_proxy',
        'hypothesis':'Observable urban morphology may supply indirect shared cues; no direct transfer benefit has been established.',
        'important_difference':'Different measured phenomena and raw units; geographic correlation is not expert predictive skill.',
        'verified_empirical_transfer':False}

def expert_cards(target,source_tasks,spec,with_relation):
    if target not in spec or target in source_tasks is False:pass
    experts=tuple(sorted(t for t in source_tasks if t!=target))
    if len(experts)!=4 or target in experts:raise ValueError('Expected four non-own source experts')
    out=[]
    for task in experts:
        card={'expert_id':task,'trained_for_indicator':task,
              'definition':spec[task]['definition'],
              'measurement_context':spec[task]['measurement_context'],
              'raw_unit':spec[task]['raw_unit'],
              'exact_formula':spec[task]['exact_formula'],
              'label_transform':spec[task]['label_transform']}
        if with_relation:card['relation_to_target']=relationship(target,task,spec)
        out.append(card)
    return out

def valid_spec_for_pilot(task,spec):
    if task not in spec or not spec[task]['definition']:raise ValueError('Unknown target')
    return spec[task]
