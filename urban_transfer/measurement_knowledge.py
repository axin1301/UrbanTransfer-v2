"""Measurement attributes derived from audited, label-free task definitions.

These are *design priors*, not learned label statistics. Unknown measurement
details stay unknown and must not be filled in using heldout labels.
"""
from .protocol import TASK_DESCRIPTIONS

# Coarse urban phenomenon and its measurement mechanism are separate.
# "accessibility" tasks share a population-normalized distance-decay protocol.
# Telecom modernization TMI is NOT assumed to be a tower count.
MEASUREMENT_ATTRIBUTES={
    'population_count': ('population','resident_count'),
    'nighttime_light_intensity': ('remote_sensing','night_radiance'),
    'education_accessibility': ('accessibility','population_normalized_distance_decay'),
    'food_accessibility': ('accessibility','population_normalized_distance_decay'),
    'healthcare_accessibility': ('accessibility','population_normalized_distance_decay'),
    'mobility_opportunity': ('accessibility','population_normalized_distance_decay'),
    'social_opportunity': ('accessibility','population_normalized_distance_decay'),
    'digital_modernization_tmi': ('telecommunications','modernization_index_uncertain'),
    'digital_total_cell_intensity': ('telecommunications','observed_base_station_intensity'),
    'digital_lte_intensity': ('telecommunications','technology_specific_base_station_intensity'),
}
if set(MEASUREMENT_ATTRIBUTES)!=set(TASK_DESCRIPTIONS):
    raise ValueError('Measurement prior must cover each known indicator')

def relatedness(target,source):
    """Purely descriptor-based compatibility, with no access to labels."""
    domain_t,mechanism_t=MEASUREMENT_ATTRIBUTES[target]
    domain_s,mechanism_s=MEASUREMENT_ATTRIBUTES[source]
    if target==source:raise ValueError('Target cannot route to itself in LOTO')
    return float(domain_t==domain_s),float(mechanism_t==mechanism_s)
