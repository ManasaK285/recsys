import pandas as pd
import streamlit as st
from src.flowfix.agent import repair_case
from src.flowfix.benchmarks import CASES
from src.flowfix.evaluation import run_benchmark
from src.flowfix.storage import save_result, recent_runs, init_db

st.set_page_config(page_title="FlowFix", page_icon="🛠️", layout="wide")
st.title("🛠️ FlowFix")
st.caption("Low-latency, test-validated program repair — reproducible offline demo")
init_db()
st.warning("Controlled benchmark only. A temporary folder is not a security sandbox; do not execute untrusted repository code on your host.")

tab_repair, tab_eval, tab_history, tab_api = st.tabs(["Repair a failure","Benchmark","Run history","API"])
with tab_repair:
    options = {f"{v['title']} ({k})":k for k,v in CASES.items()}
    label = st.selectbox("Select seeded failure", list(options))
    case_id = options[label]
    st.write(CASES[case_id]["description"])
    with st.expander("Original buggy source", expanded=True):
        st.code(CASES[case_id]["source"], language="python")
    if st.button("Run repair agent", type="primary"):
        with st.spinner("Reproducing failure, proposing patch, validating tests..."):
            result = repair_case(case_id)
            save_result(result)
        st.session_state["last_result"] = result
    result = st.session_state.get("last_result")
    if result and result.get("case_id") == case_id:
        c1,c2,c3 = st.columns(3)
        c1.metric("Status",result["status"].replace("_"," ").title())
        c2.metric("Latency",f"{result['elapsed_ms']:.1f} ms")
        c3.metric("Attempts",result["attempts"])
        st.info(result["message"])
        if result["patch"]:
            st.code(result["patch"],language="diff")
        st.text_area("Validation output",result["test_output"],height=180)
with tab_eval:
    st.write("Runs all four deterministic benchmark cases. Metrics describe only this small suite.")
    if st.button("Run complete benchmark"):
        with st.spinner("Running benchmark..."):
            report = run_benchmark()
            for row in report["cases"]:
                save_result(row)
            st.session_state["benchmark"] = report
    report = st.session_state.get("benchmark")
    if report:
        s = report["summary"]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Cases",s["total_cases"])
        c2.metric("Verified suggestions",s["verified_suggestions"])
        c3.metric("Median latency",f"{s['median_latency_ms']:.1f} ms")
        c4.metric("P95 latency",f"{s['p95_latency_ms']:.1f} ms")
        st.dataframe(pd.DataFrame(report["cases"]),use_container_width=True)
        st.caption(s["note"])
with tab_history:
    rows = recent_runs()
    if rows:
        st.dataframe(pd.DataFrame([{k:r[k] for k in ("id","case_id","status","elapsed_ms","attempts","created_at")} for r in rows]),use_container_width=True)
    else:
        st.info("No repair runs yet.")
with tab_api:
    st.markdown("Start the API in another terminal:")
    st.code("uvicorn api:app --reload",language="powershell")
    st.markdown("Open `http://127.0.0.1:8000/docs`.")
    st.code('$body = @{ case_id = "average_off_by_one" } | ConvertTo-Json\nInvoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/repair -ContentType "application/json" -Body $body',language="powershell")
