from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
DATASET_REPO = "neurips-dataset-1211/DIVE"
TARGET_DEFAULT = "how_harmful_to_you"
DEMOGRAPHIC_COLUMNS = [
    "rater_ethnicity", "rater_gender", "rater_age_group",
    "rater_country_of_birth", "rater_country_of_residence",
    "rater_nationality", "rater_employment_status"
]
MIN_GROUP_SIZE = 20
