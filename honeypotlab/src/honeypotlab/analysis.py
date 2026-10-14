"""Summary statistics for benchmark runs."""
import pandas as pd


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    required = {"condition", "agent", "task_success", "constraint_violation",
                "opportunity_action", "evaluation_awareness"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = (
        df.groupby(["condition", "agent"], as_index=False)
        .agg(
            runs=("trial", "count"),
            task_success_rate=("task_success", "mean"),
            constraint_violation_rate=("constraint_violation", "mean"),
            opportunity_action_rate=("opportunity_action", "mean"),
            evaluation_awareness_rate=("evaluation_awareness", "mean"),
        )
    )
    rate_cols = [c for c in result.columns if c.endswith("_rate")]
    result[rate_cols] = (result[rate_cols] * 100).round(2)
    return result
