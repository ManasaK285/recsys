"""Hypothesis tests for the behavioral analysis layer."""
import pandas as pd
from scipy import stats


def chi_square_test(df: pd.DataFrame, group_col: str, outcome_col: str = "decision") -> dict:
    contingency = pd.crosstab(df[group_col], df[outcome_col])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {
        "test": "chi_square_independence",
        "group_col": group_col,
        "outcome_col": outcome_col,
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "significant_at_0.05": bool(p < 0.05),
        "contingency_table": contingency.to_dict(),
    }


def anova_response_time(df: pd.DataFrame, group_col: str) -> dict:
    groups = [g["response_time_ms"].dropna().values for _, g in df.groupby(group_col)]
    groups = [g for g in groups if len(g) > 1]
    f_stat, p = stats.f_oneway(*groups)
    return {
        "test": "one_way_anova_response_time",
        "group_col": group_col,
        "f_statistic": float(f_stat),
        "p_value": float(p),
        "significant_at_0.05": bool(p < 0.05),
    }


def anova_trust(df: pd.DataFrame, group_col: str, trust_col: str = "app_trust") -> dict:
    groups = [g[trust_col].dropna().values for _, g in df.groupby(group_col)]
    groups = [g for g in groups if len(g) > 1]
    f_stat, p = stats.f_oneway(*groups)
    return {
        "test": f"one_way_anova_{trust_col}",
        "group_col": group_col,
        "f_statistic": float(f_stat),
        "p_value": float(p),
        "significant_at_0.05": bool(p < 0.05),
    }


def run_all_tests(df: pd.DataFrame) -> dict:
    return {
        "grant_vs_reason_type": chi_square_test(df, "reason_type"),
        "grant_vs_app_type": chi_square_test(df, "app_type"),
        "response_time_by_reason_type": anova_response_time(df, "reason_type"),
        "app_trust_by_reason_type": anova_trust(df, "reason_type", "app_trust"),
        "privacy_concern_by_reason_type": anova_trust(df, "reason_type", "privacy_concern"),
    }
