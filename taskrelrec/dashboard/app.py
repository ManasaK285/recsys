from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/"artifacts"
st.set_page_config(page_title="TaskRelRec v2",layout="wide")
st.title("TaskRelRec v2 — Learned Task Relationships")
a,b,c,d=st.tabs(["Overview","Relationships","Pointwise","Ranking"])

with a:
    st.write("Held-out user-level evaluation with Shared Bottom, MMoE and TaskRelRec.")
    cols=st.columns(3)
    for j,n in enumerate(["train.csv","val.csv","test.csv"]):
        p=ART/n
        if p.exists(): cols[j].metric(n[:-4].title(),f"{len(pd.read_csv(p)):,} rows")

with b:
    p=ART/"taskrelrec_relationships.csv"
    if p.exists():
        x=pd.read_csv(p,index_col=0)
        st.plotly_chart(px.imshow(x,text_auto=".2f",aspect="auto",title="Learned Relationship Matrix"),use_container_width=True)
    p=ART/"empirical_relationships.csv"
    if p.exists(): st.dataframe(pd.read_csv(p,index_col=0),use_container_width=True)

with c:
    p=ART/"pointwise_metrics.csv"
    if p.exists():
        x=pd.read_csv(p); st.dataframe(x,use_container_width=True)
        st.plotly_chart(px.bar(x,x="task",y="auc",color="model",barmode="group",title="Test ROC-AUC"),use_container_width=True)

with d:
    p=ART/"ranking_metrics.csv"
    if p.exists():
        x=pd.read_csv(p); st.dataframe(x,use_container_width=True)
        st.plotly_chart(px.bar(x,x="task",y="ndcg@10",color="model",barmode="group",title="Test NDCG@10"),use_container_width=True)
