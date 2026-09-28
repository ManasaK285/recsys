"""
ReasonLens Research Dashboard.

Reads the artifacts produced by `python -m experiments.run_all`:
  data/results/results.json      -- descriptive stats, hypothesis tests, regressions, ML metrics
  data/results/figures/*.png     -- static plots
  data/processed/tidy.csv        -- row-level joined data, for replay/drilldown

Run with:
    streamlit run dashboard/app.py
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_PATH = os.path.join(ROOT, "data", "results", "results.json")
TIDY_PATH = os.path.join(ROOT, "data", "processed", "tidy.csv")
FIGURES_DIR = os.path.join(ROOT, "data", "results", "figures")

st.set_page_config(page_title="ReasonLens Research Dashboard", layout="wide")


@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_tidy():
    if not os.path.exists(TIDY_PATH):
        return None
    return pd.read_csv(TIDY_PATH)


results = load_results()
df = load_tidy()

st.title("ReasonLens Research Dashboard")

if results is None or df is None:
    st.error(
        "No results found. Run `python -m experiments.run_all` first to generate "
        "data/results/results.json and data/processed/tidy.csv."
    )
    st.stop()

mode = results.get("mode", "unknown")
if mode == "simulation":
    st.warning(
        "SIMULATION MODE — this dashboard is showing reproducible synthetic data "
        "generated for development/testing, not real participant data.",
        icon="⚠️",
    )
else:
    st.success("PILOT MODE — showing real participant data collected via the Android app.", icon="✅")

st.caption(f"Results generated at {results.get('generated_at', 'unknown time')}")

tabs = st.tabs(
    [
        "Overview",
        "Permission Behavior",
        "App Type Effects",
        "Reason Effects",
        "Precision Choice",
        "Trust & Perception",
        "Statistical Analysis",
        "ML / Explainability",
        "Individual Experiment Replay",
    ]
)

# ---------------- Overview ----------------
with tabs[0]:
    st.subheader("Overview")
    overall = results["descriptive"]["overall"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Participants", overall["n"])
    c2.metric("Grant rate", f"{overall['grant_rate']:.1%}")
    c3.metric("Denial rate", f"{overall['denial_rate']:.1%}")
    if overall["precise_rate_of_granted"] is not None:
        c4.metric("Precise (of granted)", f"{overall['precise_rate_of_granted']:.1%}")
    st.markdown("---")
    st.markdown(
        "**Conditions:** app type × reason type, randomized and assigned deterministically "
        "per participant (see `experiments/randomization.py`)."
    )
    st.dataframe(df[["app_type", "reason_type", "decision", "precision"]].head(20))

# ---------------- Permission Behavior ----------------
with tabs[1]:
    st.subheader("Permission Behavior")
    fig_path = os.path.join(FIGURES_DIR, "grant_rate_by_reason.png")
    if os.path.exists(fig_path):
        st.image(fig_path, use_container_width=True)
    st.dataframe(pd.DataFrame(results["descriptive"]["by_reason_type"]))

# ---------------- App Type Effects ----------------
with tabs[2]:
    st.subheader("App Type Effects")
    fig_path = os.path.join(FIGURES_DIR, "grant_rate_by_app.png")
    if os.path.exists(fig_path):
        st.image(fig_path, use_container_width=True)
    st.dataframe(pd.DataFrame(results["descriptive"]["by_app_type"]))

# ---------------- Reason Effects ----------------
with tabs[3]:
    st.subheader("Reason x App Interaction")
    fig_path = os.path.join(FIGURES_DIR, "heatmap_reason_app.png")
    if os.path.exists(fig_path):
        st.image(fig_path, use_container_width=True)
    st.dataframe(pd.DataFrame(results["descriptive"]["by_reason_and_app"]))

# ---------------- Precision Choice ----------------
with tabs[4]:
    st.subheader("Precision Choice (approximate vs. precise, given grant)")
    granted = df[df["decision"] == "granted"]
    if len(granted):
        precision_rate = granted.groupby("reason_type")["precise"].mean().reset_index()
        st.bar_chart(precision_rate.set_index("reason_type"))
    else:
        st.info("No granted decisions in this dataset.")

# ---------------- Trust & Perception ----------------
with tabs[5]:
    st.subheader("Trust & Perception")
    fig_path = os.path.join(FIGURES_DIR, "trust_by_reason.png")
    if os.path.exists(fig_path):
        st.image(fig_path, use_container_width=True)
    fig_path2 = os.path.join(FIGURES_DIR, "response_time_by_reason.png")
    if os.path.exists(fig_path2):
        st.image(fig_path2, use_container_width=True)

# ---------------- Statistical Analysis ----------------
with tabs[6]:
    st.subheader("Hypothesis Tests")
    st.json(results["hypothesis_tests"])

    st.subheader("Logistic Regression: Main Effects")
    st.dataframe(pd.DataFrame(results["regressions"]["main_effects"]["coefficients"]))
    st.caption(
        f"Pseudo R²: {results['regressions']['main_effects']['pseudo_r2']:.4f} | "
        f"AIC: {results['regressions']['main_effects']['aic']:.1f} | "
        f"N: {results['regressions']['main_effects']['n_obs']}"
    )

    st.subheader("Logistic Regression: Interaction Model")
    st.dataframe(pd.DataFrame(results["regressions"]["interaction"]["coefficients"]))
    st.caption(
        f"Pseudo R²: {results['regressions']['interaction']['pseudo_r2']:.4f} | "
        f"AIC: {results['regressions']['interaction']['aic']:.1f}"
    )

# ---------------- ML / Explainability ----------------
with tabs[7]:
    st.subheader("Reason-Quality ML Model")
    ml = results["ml_metrics"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Reason-type classifier (from text)**")
        st.json(ml["classifier"])
    with c2:
        st.markdown("**Quality-score regressor (from text)**")
        st.json(ml["regressor"])

    st.markdown("**Explainability**")
    expl = ml.get("explainability", {})
    if "regressor_top_terms" in expl:
        st.markdown("*Top terms increasing / decreasing predicted quality:*")
        st.json(expl["regressor_top_terms"])
    if "shap" in expl:
        st.markdown("*SHAP summary (best-effort):*")
        st.json(expl["shap"])

# ---------------- Individual Experiment Replay ----------------
with tabs[8]:
    st.subheader("Individual Experiment Replay")
    participant_ids = df["participant_id"].dropna().unique().tolist()
    selected = st.selectbox("Select a participant_id", participant_ids[:500])
    if selected:
        row = df[df["participant_id"] == selected].iloc[0]
        st.json(row.to_dict())
