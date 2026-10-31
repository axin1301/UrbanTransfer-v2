"""Auditable text-only knowledge profiles for urban indicator transfer.

Potential drivers are hypotheses, not ground-truth correlations. Never include
test-label moments, test performance or target city statistics in profiles.
"""
from .protocol import TASK_DESCRIPTIONS

PROFILES = {
'population_count': ('Resident population magnitude within a geographic tile.',
    'Counts people rather than facilities; spatial concentration is affected by residential land use.',
    'Built-up housing density, accessibility and settlement form may correlate.'),
'nighttime_light_intensity': ('Remotely observed nighttime luminosity intensity within the tile.',
    'Measures emitted/observed night light, not population or economic activity directly.',
    'Illuminated built-up areas, commercial activity and infrastructure may correlate.'),
'education_accessibility': ('Accessibility to education facilities, normalized by population.',
    'The dataset specifies a distance-decay accessibility calculation; this is not a simple school count.',
    'Residential distribution, transport access, and school location may matter.'),
'food_accessibility': ('Accessibility to food facilities, normalized by population.',
    'The dataset specifies a distance-decay accessibility calculation; this is not a raw shop count.',
    'Residential distribution, retail centers, and transport access may matter.'),
'digital_modernization_tmi': ('Cellular infrastructure modernization indicator within the tile.',
    'Precise TMI construction requires a dataset-methodology check; do not equate it to total tower counts.',
    'Infrastructure generation mix and deployment patterns may matter.'),
'digital_total_cell_intensity': ('Intensity of all observed cellular base stations within the tile.',
    'Includes the observed cellular infrastructure aggregate, potentially including multiple generations.',
    'Urban demand, land use and network deployment strategies may matter.'),
'digital_lte_intensity': ('Intensity of LTE or fourth-generation cellular network infrastructure.',
    'LTE is a technology-specific component of cellular infrastructure, not necessarily proportional to all cellular towers.',
    'Demand, infrastructure deployment, coverage planning and built-up patterns may matter.'),
'healthcare_accessibility': ('Accessibility to healthcare facilities, normalized by population.',
    'Distance-decay accessibility to healthcare facilities, not simply the number of clinics.',
    'Hospital location, residential distribution and local connectivity may matter.'),
'mobility_opportunity': ('Accessibility to mobility facilities, normalized by population.',
    'Uses population-normalized distance-decay accessibility; precise facility taxonomy requires dataset-methodology confirmation.',
    'Transit availability, population distribution and connectivity may matter.'),
'social_opportunity': ('Accessibility to social facilities, normalized by population.',
    'Uses population-normalized distance-decay accessibility; precise facility taxonomy requires dataset-methodology confirmation.',
    'Social amenities, residential distribution and travel accessibility may matter.'),
}

def task_texts(profile='short', reviewed_llm_file=None):
    if profile == 'short':
        return dict(TASK_DESCRIPTIONS)
    if profile == 'structured':
        return {task: (
            f'Urban indicator: {task}.\n'
            f'Definition: {definition}\n'
            f'Measurement: {mechanism}\n'
            f'Possible spatial drivers (hypotheses only): {drivers}\n'
            'Output: nonnegative indicator, modeled as log1p-transformed value. '
            'Do not infer absolute label means or ranges from this description.'
        ) for task,(definition,mechanism,drivers) in PROFILES.items()}
    if profile == 'llm':
        if not reviewed_llm_file:
            raise ValueError('Profile llm requires --llm-profile-file containing reviewed task descriptions')
        import json
        from pathlib import Path
        data=json.loads(Path(reviewed_llm_file).read_text())
        if set(data)!=set(TASK_DESCRIPTIONS) or any(not isinstance(v,str) or len(v.strip())<50 for v in data.values()):
            raise ValueError('Reviewed LLM profile JSON must contain exactly ten nonempty long task descriptions')
        return data
    raise ValueError(f'Unknown profile: {profile}')
