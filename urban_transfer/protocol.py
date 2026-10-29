"""Evaluation protocol. Target test labels are never used for training/model selection."""
SOURCE_TASKS = ("population_count", "nighttime_light_intensity", "education_accessibility", "food_accessibility", "digital_modernization_tmi", "digital_total_cell_intensity")
TARGET_TASKS = ("healthcare_accessibility", "mobility_opportunity", "social_opportunity", "digital_lte_intensity")
TASK_DESCRIPTIONS = {
    "population_count": "Population count in the local geographic tile.",
    "nighttime_light_intensity": "Annual intensity of nighttime luminosity within the tile.",
    "education_accessibility": "Population-normalized distance-decay accessibility to education facilities.",
    "food_accessibility": "Population-normalized distance-decay accessibility to food facilities.",
    "digital_modernization_tmi": "Spatial intensity of cellular network infrastructure.",
    "digital_total_cell_intensity": "Spatial intensity of all observed cellular base stations.",
    "digital_lte_intensity": "Spatial intensity of LTE cellular network infrastructure.",
    "healthcare_accessibility": "Population-normalized distance-decay accessibility to healthcare facilities.",
    "mobility_opportunity": "Population-normalized distance-decay accessibility to mobility facilities.",
    "social_opportunity": "Population-normalized distance-decay accessibility to social facilities.",
}
