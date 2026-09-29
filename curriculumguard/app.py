"""streamlit run app.py"""
import pandas as pd
import plotly.express as px
import streamlit as st
from cg import governance as gv
from cg.pipeline import load_or_generate, run_all
from cg.report import build_report

st.set_page_config(page_title="CurriculumGuard", layout="wide")
st.title("CurriculumGuard: who shapes the AI curriculum policy?")
up = st.sidebar.file_uploader("Feedback CSV (group,stance,text,severity[,subgroup])", type="csv")
k = st.sidebar.slider("NLP clusters", 3, 8, 5)
df0 = pd.read_csv(up) if up else load_or_generate()
if not up:
    st.sidebar.warning("Using SYNTHETIC demo data.")


@st.cache_data(show_spinner="Analyzing...")
def analyze(d, k):
    return run_all(d, k=k)


res = analyze(df0, k)
df = res["df"]
tabs = st.tabs(["Representation", "Concerns & clusters", "Majority vs balanced", "Counterfactual", "Policies", "Evidence", "Report"])

with tabs[0]:
    a = res["audit"]
    st.plotly_chart(px.bar(a.melt("group", ["participation", "population"]), x="group", y="value", color="variable",
                           barmode="group", title="Participation vs population share"))
    st.dataframe(a); st.caption("Student subgroups (n<5 suppressed)"); st.dataframe(res["subgroup"])
    st.dataframe(res["power"].head(12))
with tabs[1]:
    cm = res["concern_by_group"]
    st.plotly_chart(px.imshow(cm, text_auto=".2f", aspect="auto", title="Concern prevalence by group"))
    c = res["clusters"]
    st.write(f"Clustering ({c['method']}): silhouette **{c['silhouette']:.2f}**, seed stability (ARI) **{c['stability_ari']:.2f}**")
    st.table(pd.DataFrame({"cluster": list(c["terms"]), "top terms": [", ".join(v) for v in c["terms"].values()],
                           "size": [(df["cluster"] == i).sum() for i in c["terms"]]}))
    st.caption("Extractor vs gold labels (synthetic data only)"); st.dataframe(res["extractor_eval"])
with tabs[2]:
    s = res["schemes"]
    st.plotly_chart(px.bar(s.melt("scheme", gv.POLICIES), x="scheme", y="value", color="variable", barmode="group"))
    st.dataframe(s)
    st.caption("Concern retention (share of a group's top-3 concerns kept in aggregate top-3)")
    st.dataframe(res["retention"])
    sch = st.selectbox("Bootstrap CI for scheme", list(res["boot"]))
    st.dataframe(res["boot"][sch])
with tabs[3]:
    g = st.selectbox("Group", list(res["sweeps"]))
    cur = float((df["group"] == g).mean())
    share = st.slider(f"{g} share of participants", 0.0, 1.0, round(cur, 2), 0.01)
    cf = gv.counterfactual(df, g, share)
    st.metric("Winning policy", cf["winner"], help=f"current share {cur:.0%}")
    sw = res["sweeps"][g].melt(["group", "share", "winner"], gv.POLICIES)
    st.plotly_chart(px.line(sw, x="share", y="value", color="variable", title=f"Policy support as {g} share varies"))
    st.dataframe(res["flips"])
with tabs[4]:
    st.write("Exposure values are editable assumptions in `cg/policy.py`. No policy is declared best.")
    st.plotly_chart(px.bar(res["policy"], x="policy", y="unaddressed_burden", color="scheme", barmode="group"))
    st.dataframe(res["policy"])
with tabs[5]:
    q = st.text_input("Search evidence", "student data privacy")
    for r in res["retriever"].search(q, 3):
        st.markdown(f"**[{r['source']}]** (score {r['score']:.2f})  \n{r['text']}")
    hits, rate = res["retriever"].evaluate()
    st.caption(f"Retrieval hit@2 on concern queries: {rate:.0%}")
with tabs[6]:
    md = build_report(res)
    st.markdown(md)
    st.download_button("Download report.md", md, "report.md")
