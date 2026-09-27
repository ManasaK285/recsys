"""
The primary research analysis: logistic regression of permission-grant
decisions on reason_type and app_type (main effects and interaction),
reported with confidence intervals and odds ratios (effect sizes).
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _fit_summary(model_result, label: str) -> dict:
    params = model_result.params
    conf = model_result.conf_int()
    conf.columns = ["ci_lower", "ci_upper"]
    odds_ratios = np.exp(params)
    or_conf = np.exp(conf)

    coef_table = []
    for name in params.index:
        coef_table.append(
            {
                "term": name,
                "coef": float(params[name]),
                "p_value": float(model_result.pvalues[name]),
                "odds_ratio": float(odds_ratios[name]),
                "or_ci_lower": float(or_conf.loc[name, "ci_lower"]),
                "or_ci_upper": float(or_conf.loc[name, "ci_upper"]),
            }
        )

    return {
        "model": label,
        "n_obs": int(model_result.nobs),
        "pseudo_r2": float(model_result.prsquared) if hasattr(model_result, "prsquared") else None,
        "log_likelihood": float(model_result.llf),
        "aic": float(model_result.aic),
        "coefficients": coef_table,
    }


def logistic_main_effects(df: pd.DataFrame) -> dict:
    data = df.copy()
    data["reason_type"] = pd.Categorical(data["reason_type"], categories=sorted(data["reason_type"].unique()))
    data["app_type"] = pd.Categorical(data["app_type"], categories=sorted(data["app_type"].unique()))
    model = smf.logit("granted ~ C(reason_type) + C(app_type)", data=data).fit(disp=0)
    return _fit_summary(model, "granted ~ reason_type + app_type (main effects)")


def logistic_interaction(df: pd.DataFrame) -> dict:
    data = df.copy()
    data["reason_type"] = pd.Categorical(data["reason_type"], categories=sorted(data["reason_type"].unique()))
    data["app_type"] = pd.Categorical(data["app_type"], categories=sorted(data["app_type"].unique()))
    model = smf.logit("granted ~ C(reason_type) * C(app_type)", data=data).fit(disp=0)
    return _fit_summary(model, "granted ~ reason_type * app_type (interaction)")


def run_all_regressions(df: pd.DataFrame) -> dict:
    return {
        "main_effects": logistic_main_effects(df),
        "interaction": logistic_interaction(df),
    }
