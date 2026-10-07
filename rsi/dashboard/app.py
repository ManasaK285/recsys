import json
import sqlite3
import pandas as pd
import streamlit as st
from src.config import DB_PATH

st.set_page_config(page_title="DreamAlgo-RSI", layout="wide")
st.title("DreamAlgo-RSI")
st.caption("Discovery trees → replay worlds → offline policy dreaming → recursive improvement")

if not DB_PATH.exists():
    st.info("Run `python scripts/run_demo.py` first.")
    st.stop()

con = sqlite3.connect(DB_PATH)

runs = pd.read_sql_query("SELECT * FROM runs ORDER BY created_at DESC", con)
nodes = pd.read_sql_query("SELECT * FROM nodes ORDER BY created_at DESC", con)
pe = pd.read_sql_query("SELECT * FROM policy_evaluations ORDER BY id DESC", con)

c1, c2, c3 = st.columns(3)
c1.metric("Runs", len(runs))
c2.metric("Discovery nodes", len(nodes))
c3.metric("Dream evaluations", len(pe))

if not nodes.empty:
    rows = []
    for _, r in nodes.iterrows():
        ev = json.loads(r["evaluation_json"])
        rows.append({
            "node": r["node_id"][:8],
            "approach": r["approach"],
            "depth": r["depth"],
            "score": ev["score"],
            "runtime_ms": ev["runtime_ms"],
            "correct": ev["correct"],
            "policy": r["policy_version"],
        })
    df = pd.DataFrame(rows)
    st.subheader("Discovery tree nodes")
    st.dataframe(df, use_container_width=True)

    st.subheader("Score by depth")
    st.line_chart(df.groupby("depth")["score"].max())

if not pe.empty:
    st.subheader("Offline dreaming")
    st.dataframe(pe[["policy_id", "world_id", "reward", "work", "created_at"]],
                 use_container_width=True)
