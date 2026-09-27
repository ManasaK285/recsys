"""
Loads the raw per-table data (from data/synthetic or a real pilot export)
and joins it into a single tidy dataframe: one row per participant, with
their assignment, permission decision, and survey responses attached.
"""
import os
import pandas as pd


def load_raw(data_dir: str):
    participants = pd.read_csv(os.path.join(data_dir, "participants.csv"))
    assignments = pd.read_csv(os.path.join(data_dir, "assignments.csv"))
    events = pd.read_csv(os.path.join(data_dir, "permission_events.csv"))
    surveys = pd.read_csv(os.path.join(data_dir, "survey_responses.csv"))
    return participants, assignments, events, surveys


def build_tidy_dataframe(data_dir: str) -> pd.DataFrame:
    participants, assignments, events, surveys = load_raw(data_dir)

    df = participants.merge(assignments, on="participant_id", how="left")
    df = df.merge(events, on="participant_id", how="left")
    df = df.merge(surveys, on="participant_id", how="left")

    df["granted"] = (df["decision"] == "granted").astype(int)
    df["precise"] = (df["precision"] == "precise").astype(int)
    df["precise"] = df["precise"].where(df["decision"] == "granted", other=pd.NA)

    return df


def validate(df: pd.DataFrame) -> list:
    """Basic sanity checks; returns a list of warning strings (empty if clean)."""
    warnings = []
    required = ["participant_id", "app_type", "reason_type", "decision"]
    for col in required:
        if col not in df.columns:
            warnings.append(f"Missing required column: {col}")
    if "decision" in df.columns:
        bad = ~df["decision"].isin(["granted", "denied"]) & df["decision"].notna()
        if bad.any():
            warnings.append(f"{bad.sum()} rows have an invalid 'decision' value")
    if df["participant_id"].duplicated().any():
        warnings.append("Duplicate participant_id rows found after join (unexpected 1:many merge)")
    return warnings


def save_processed(df: pd.DataFrame, processed_dir: str, name: str = "tidy.csv"):
    os.makedirs(processed_dir, exist_ok=True)
    path = os.path.join(processed_dir, name)
    df.to_csv(path, index=False)
    return path
