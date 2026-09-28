"""Descriptive statistics: grant/denial/precision rates, overall and by group."""
import pandas as pd


def overall_rates(df: pd.DataFrame) -> dict:
    n = len(df)
    granted = df["granted"].sum()
    denied = n - granted
    precise_of_granted = df.loc[df["granted"] == 1, "precise"]
    return {
        "n": int(n),
        "grant_rate": float(granted / n) if n else None,
        "denial_rate": float(denied / n) if n else None,
        "precise_rate_of_granted": float(precise_of_granted.mean()) if len(precise_of_granted) else None,
        "approximate_rate_of_granted": float(1 - precise_of_granted.mean()) if len(precise_of_granted) else None,
    }


def rate_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = df.groupby(group_col, dropna=False).agg(
        n=("participant_id", "count"),
        grant_rate=("granted", "mean"),
        precise_rate_of_granted=("precise", "mean"),
        mean_response_time_ms=("response_time_ms", "mean"),
        mean_app_trust=("app_trust", "mean"),
        mean_android_trust=("android_trust", "mean"),
        mean_necessity=("necessity", "mean"),
        mean_privacy_concern=("privacy_concern", "mean"),
    ).reset_index()
    return g


def rate_by_reason_and_app(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["app_type", "reason_type"], dropna=False).agg(
        n=("participant_id", "count"),
        grant_rate=("granted", "mean"),
        precise_rate_of_granted=("precise", "mean"),
    ).reset_index()
    return g


def summarize(df: pd.DataFrame) -> dict:
    return {
        "overall": overall_rates(df),
        "by_reason_type": rate_by_group(df, "reason_type").to_dict(orient="records"),
        "by_app_type": rate_by_group(df, "app_type").to_dict(orient="records"),
        "by_reason_and_app": rate_by_reason_and_app(df).to_dict(orient="records"),
    }
