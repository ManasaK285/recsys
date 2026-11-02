from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from src.diveguard.config import ARTIFACT_DIR, TARGET_DEFAULT, DEMOGRAPHIC_COLUMNS, MIN_GROUP_SIZE
from src.diveguard.data import load_processed, load_raw, clean_data
from src.diveguard.demo import demo_data
from src.diveguard.inference import predict

st.set_page_config(page_title="DIVEGuard AI", page_icon="🧭", layout="wide")
st.markdown("""
<style>
.main {background: #0b1020;}
.block-container {padding-top: 1.5rem; max-width: 1500px;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid #273653; border-radius: 18px;
background: linear-gradient(120deg,#172554,#111827 60%,#312e81); margin-bottom: 1rem;}
.hero h1 {color: white; margin: 0;}
.hero p {color: #cbd5e1; margin: .4rem 0 0 0;}
.small-note {color: #94a3b8; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    processed = load_processed(TARGET_DEFAULT)
    if processed is not None:
        return processed, False
    try:
        raw = load_raw()
        return clean_data(raw, TARGET_DEFAULT), False
    except Exception:
        return demo_data(), True

df, demo_mode = get_data()
target = TARGET_DEFAULT if TARGET_DEFAULT in df.columns else "how_harmful_to_you"
st.markdown('<div class="hero"><h1>🧭 DIVEGuard AI</h1><p>Pluralistic safety intelligence · Understand disagreement, not just averages.</p></div>', unsafe_allow_html=True)
if demo_mode:
    st.warning("Demo mode: synthetic data is being shown because the DIVE dataset is not present locally. These examples are not research results. Download the real data to run the audit.")
else:
    st.success("Real DIVE annotation data loaded.")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Workspace", ["Overview", "Pluralism Audit", "Model Lab", "Perspective Explorer", "Disagreement Explorer"])
st.sidebar.caption("DIVE · NeurIPS 2025 inspiration")
st.sidebar.markdown("[Dataset](https://huggingface.co/datasets/neurips-dataset-1211/DIVE) · [Paper](https://arxiv.org/abs/2507.13383)")

ratings = pd.to_numeric(df[target], errors="coerce").dropna()
if page == "Overview":
    st.subheader("Dataset overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Annotations", f"{len(df):,}")
    c2.metric("Unique prompts/images", f"{df['image_id'].nunique():,}" if "image_id" in df else "—")
    c3.metric("Mean harm rating", f"{ratings.mean():.2f}/4")
    c4.metric("Topics", f"{df['topic'].nunique():,}" if "topic" in df else "—")
    a,b = st.columns(2)
    with a:
        st.plotly_chart(px.histogram(df, x=target, nbins=5, title="Harmfulness rating distribution",
                                     color_discrete_sequence=["#818cf8"]), use_container_width=True)
    with b:
        topic = df.groupby("topic", dropna=False)[target].mean().sort_values().reset_index()
        st.plotly_chart(px.bar(topic, x=target, y="topic", orientation="h",
                               title="Average rating by topic", color=target,
                               color_continuous_scale="Viridis"), use_container_width=True)
    st.caption("Ratings are human judgments in this dataset, not objective ground truth.")

elif page == "Pluralism Audit":
    st.subheader("Pluralism audit")
    st.write("Compare aggregate ratings across available rater attributes. Groups are suppressed below the minimum sample size.")
    attrs = [c for c in DEMOGRAPHIC_COLUMNS if c in df.columns]
    if not attrs:
        st.info("No demographic columns found in this data file.")
    else:
        attr = st.selectbox("Audit attribute", attrs)
        agg = df.groupby(attr)[target].agg(["mean","count","std"]).reset_index()
        agg = agg[agg["count"] >= MIN_GROUP_SIZE]
        if agg.empty:
            st.info(f"No group has at least {MIN_GROUP_SIZE} rows in this selection.")
        else:
            st.plotly_chart(px.bar(agg, x=attr, y="mean", error_y=agg["std"].fillna(0),
                                   hover_data=["count"], title=f"Mean rating by {attr}",
                                   color="mean", color_continuous_scale="Viridis"), use_container_width=True)
            st.dataframe(agg.rename(columns={"mean":"mean_rating","count":"annotations","std":"std_dev"}),
                         use_container_width=True, hide_index=True)
            st.caption("Aggregate differences do not imply that every member of a group shares the same view.")

elif page == "Model Lab":
    st.subheader("Global vs pluralistic model")
    metric_file = ARTIFACT_DIR / f"metrics_{target}.json"
    if metric_file.exists():
        payload = json.loads(metric_file.read_text())
        st.caption(f"Evaluation split: {payload.get('split')} · train={payload.get('train_rows'):,} · test={payload.get('test_rows'):,}")
        rows = []
        for name, vals in payload["models"].items():
            rows.append({"Model": name.title(), **vals})
        metrics = pd.DataFrame(rows)
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        if {"global","pluralistic"}.issubset(payload["models"]):
            m1,m2 = st.columns(2)
            m1.metric("Pluralistic MAE", f"{payload['models']['pluralistic']['mae']:.3f}",
                      delta=f"{payload['models']['pluralistic']['mae']-payload['models']['global']['mae']:+.3f} vs global (lower is better)")
            m2.metric("Pluralistic Macro-F1", f"{payload['models']['pluralistic']['macro_f1']:.3f}",
                      delta=f"{payload['models']['pluralistic']['macro_f1']-payload['models']['global']['macro_f1']:+.3f} vs global")
        group_file = ARTIFACT_DIR / f"group_metrics_{target}.csv"
        if group_file.exists():
            groups = pd.read_csv(group_file)
            if not groups.empty:
                st.plotly_chart(px.bar(groups, x="group", y="mae", color="attribute",
                                       hover_data=["n"], title="Pluralistic model subgroup MAE"),
                                use_container_width=True)
    else:
        st.info("Train the models to populate this page: `python -m src.diveguard.train`.")
        st.code("python -m src.diveguard.train", language="powershell")

elif page == "Perspective Explorer":
    st.subheader("Perspective-conditioned prediction")
    st.write("Explore how the trained model's prediction changes with selected context. These are model estimates, not claims about real people.")
    model_name = st.radio("Model", ["pluralistic", "global"], horizontal=True)
    prompt = st.text_area("Text prompt", "A community leader speaking at a town hall", height=90)
    topic_options = sorted(df["topic"].dropna().astype(str).unique().tolist())
    topic = st.selectbox("Topic", topic_options or ["unknown"])
    context = {}
    for col in [c for c in DEMOGRAPHIC_COLUMNS if c in df.columns]:
        vals = sorted(df[col].dropna().astype(str).unique().tolist())
        if vals:
            context[col] = st.selectbox(col.replace("rater_","").replace("_"," ").title(), vals, key=col+"_perspective")
    if st.button("Evaluate perspective", type="primary"):
        try:
            result = predict(prompt, topic, context, model_name=model_name, target=target)
            st.metric("Predicted rating", f"{result['predicted_rating']} / 4")
            probs = pd.DataFrame({"rating": list(result["probabilities"].keys()),
                                  "probability": list(result["probabilities"].values())})
            st.plotly_chart(px.bar(probs, x="rating", y="probability",
                                   title="Predicted rating distribution",
                                   color="probability", color_continuous_scale="Viridis"),
                            use_container_width=True)
            st.json(result)
        except Exception as exc:
            st.error(f"Model unavailable: {exc}")
            st.caption("Train models first using the command shown in Model Lab.")

else:
    st.subheader("Disagreement explorer")
    st.write("Find prompts where observed human ratings vary substantially.")
    prompt_col = "text_prompt"
    summary = df.groupby(["image_id", prompt_col, "topic"])[target].agg(
        mean="mean", std="std", count="count", min="min", max="max"
    ).reset_index()
    summary["std"] = summary["std"].fillna(0)
    summary = summary[summary["count"] >= 5].sort_values("std", ascending=False)
    if summary.empty:
        st.info("Not enough repeated annotations to compute disagreement.")
    else:
        st.dataframe(summary.head(30).rename(columns={"std":"rating_disagreement"}),
                     use_container_width=True, hide_index=True)
        st.plotly_chart(px.scatter(summary, x="mean", y="std", size="count", color="topic",
                                   hover_data=[prompt_col], title="Mean rating vs disagreement"),
                        use_container_width=True)

st.divider()
st.markdown('<p class="small-note">Research prototype · Demographics are contextual proxies, not deterministic identities. Not a production moderation system.</p>', unsafe_allow_html=True)
