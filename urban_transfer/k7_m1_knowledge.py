"""K7 M1 mechanism-level relation cards for ONE 2-source -> 1-target pilot.

Fixed: population_count + food_accessibility -> education_accessibility.
Text assertions are sourced from repository profiles; spatial proposals are
explicit hypotheses. The texts contain NO outcome labels or target moments.
"""
from urban_transfer.task_profiles import PROFILES

TRAIN_TASKS = ("population_count", "food_accessibility")
HELDOUT_TASK = "education_accessibility"

# Deliberately written as *directed transfer instructions*, not symmetric
# relatedness. Only the four edges used by the pilot are included.
EDGE_CARDS = {
    ("population_count", "food_accessibility"): {
        "shared_visual": "Residential distribution, settlement form, and built-up patterns may describe population demand around food facilities.",
        "measurement_link": "Food accessibility is population-normalized and distance-decayed per repository descriptions; population may affect its normalization, but the exact formula is not independently verified here.",
        "transferable": "Reuse spatial representation of residential concentration and settlement structure, not the absolute population regressor.",
        "nontransferable": "Population magnitude does not uniquely determine food facility supply, routing distance, or accessibility.",
        "hypothesis": "The relative utility of residential visual cues for food accessibility is an unverified hypothesis.",
    },
    ("food_accessibility", "population_count"): {
        "shared_visual": "Food-serving built-up areas and settlement patterns may provide proxies for where people live, but business locations are not residential counts.",
        "measurement_link": "Food accessibility is a population-normalized, distance-decay facility measure; population count is an absolute count. Their target units and numerical scales are not interchangeable.",
        "transferable": "Reuse generic built-environment features, not the food-accessibility output head.",
        "nontransferable": "Food facility coverage is not a direct population measurement and the relationship can vary across cities.",
        "hypothesis": "Retail/settlement co-location is plausible but not empirically established for these tiles.",
    },
    ("population_count", "education_accessibility"): {
        "shared_visual": "Housing concentration, settlement morphology and residential distribution may indicate where educational demand is located.",
        "measurement_link": "Education accessibility is described as population-normalized and distance-decayed, whereas population count measures residents; no sign, magnitude or exact formula is asserted.",
        "transferable": "Reuse population-related visual representations of residential concentration, not population output values or output-scale parameters.",
        "nontransferable": "Higher population does not imply higher education accessibility; school supply and service catchments are separate factors.",
        "hypothesis": "Population-related visual features might explain demand-side spatial variation; this is not a verified transfer benefit.",
    },
    ("food_accessibility", "education_accessibility"): {
        "shared_visual": "Road connectivity, built environment and facility catchment patterns may support both kinds of spatial accessibility.",
        "measurement_link": "Both indicators are described in the repository as population-normalized distance-decay facility accessibility, but their facility classes, parameters and numeric scales may differ.",
        "transferable": "Reuse facility-accessibility representations and spatial connectivity concepts, not food facilities as direct surrogates for schools.",
        "nontransferable": "Schools and food services have distinct location, demand and supply dynamics. Do not equate their intensity, distance decay or target moments.",
        "hypothesis": "Shared accessibility mechanics are supported by profiles; transferable visual cues remain hypotheses to test.",
    },
}


def relation_text(source, target):
    if (source, target) not in EDGE_CARDS:
        raise ValueError("Unregistered mechanism relation")
    source_definition, source_measurement, _ = PROFILES[source]
    target_definition, target_measurement, _ = PROFILES[target]
    card = EDGE_CARDS[source, target]
    return (
        "Directed transfer of urban indicator prediction knowledge.\n"
        f"Source: {source}. Definition: {source_definition}\n"
        f"Source measurement context: {source_measurement}\n"
        f"Target: {target}. Definition: {target_definition}\n"
        f"Target measurement context: {target_measurement}\n"
        f"Potential shared visual evidence (HYPOTHESIS): {card['shared_visual']}\n"
        f"Measurement link (source-profile grounding; exact formulas unverified): {card['measurement_link']}\n"
        f"Transferable representation (HYPOTHESIS): {card['transferable']}\n"
        f"Do NOT transfer: {card['nontransferable']}\n"
        f"Evidence limitation: {card['hypothesis']}\n"
        "Do not assert causal relations, exact target ranges, target means, "
        "target standard deviations, or known empirical transfer accuracy."
    )


def all_relation_texts():
    return {pair: relation_text(*pair) for pair in EDGE_CARDS}


def show():
    for (src, tgt), message in all_relation_texts().items():
        print(f"\n{'='*65}\n{src} -> {tgt}\n{'='*65}\n{message}")


if __name__ == "__main__":
    show()
