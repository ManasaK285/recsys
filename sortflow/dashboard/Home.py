import os
import requests
import pandas as pd
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000")
st.set_page_config(page_title="SortFlow Control Tower", page_icon="📦", layout="wide")
st.title("📦 SortFlow | Network Operations Control Tower")
st.caption("Synthetic logistics simulation · event-driven planning · auditability · human-approved decisions")


def get(path, fallback=None):
    try:
        r = requests.get(f"{API}{path}", timeout=5); r.raise_for_status(); return r.json()
    except Exception as exc:
        st.error(f"API unavailable: {exc}"); return fallback


def post(path, body):
    r = requests.post(f"{API}{path}", json=body, timeout=10); r.raise_for_status(); return r.json()

summary = get("/summary", {})
centers = get("/centers", [])
workflows = get("/workflows?limit=30", [])
events = get("/events?limit=30", [])
metrics = get("/metrics", {})
health = get("/health/dependencies", {})

if summary:
    cols = st.columns(5)
    cols[0].metric("Sort centers", summary.get("sort_centers", 0))
    cols[1].metric("Network demand / hr", f'{summary.get("network_demand_per_hour", 0):,}')
    cols[2].metric("Network utilization", f'{summary.get("network_utilization_pct", 0)}%')
    cols[3].metric("Completed workflows", summary.get("completed_workflows", 0))
    cols[4].metric("Failed workflows", summary.get("failed_workflows", 0))

with st.expander("Platform health", expanded=False):
    st.json(health)
    st.json(metrics)

st.divider()
tab_network, tab_event, tab_scenario, tab_history = st.tabs(["Network", "Inject event", "Scenario lab", "Workflow & audit"])
with tab_network:
    if centers:
        df = pd.DataFrame(centers)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.subheader("Center utilization")
        st.bar_chart(df.set_index("id")[["utilization_pct"]])

with tab_event:
    st.subheader("Inject an operational disruption")
    with st.form("event_form"):
        event_type = st.selectbox("Event type", ["DEMAND_SURGE", "LABOR_SHORTAGE", "CONVEYOR_FAILURE", "TRANSPORT_DELAY"])
        center_ids = [c["id"] for c in centers] or [f"SC-{i:02d}" for i in range(1, 13)]
        center_id = st.selectbox("Sort center", center_ids, index=center_ids.index("SC-08") if "SC-08" in center_ids else 0)
        severity = st.selectbox("Severity", ["LOW", "MEDIUM", "HIGH", "CRITICAL"], index=2)
        surge = st.slider("Demand increase (%)", 0, 150, 40) if event_type == "DEMAND_SURGE" else 0
        labor = st.slider("Labor reduction (%)", 0, 100, 20) if event_type == "LABOR_SHORTAGE" else 0
        capacity = st.slider("Capacity reduction (%)", 0, 100, 30) if event_type in ("CONVEYOR_FAILURE", "TRANSPORT_DELAY") else 0
        submitted = st.form_submit_button("Trigger planning workflow", type="primary")
        if submitted:
            try:
                result = post("/events", {"event_type": event_type, "center_id": center_id, "severity": severity,
                    "volume_change_pct": surge, "labor_reduction_pct": labor, "capacity_reduction_pct": capacity})
                st.success(f"Workflow {result['workflow_id']} · {result['status']}")
                st.json(result)
            except Exception as exc: st.error(f"Could not create event: {exc}")

with tab_scenario:
    st.subheader("Network-wide what-if analysis")
    with st.form("scenario_form"):
        multiplier = st.slider("Demand multiplier", 1.0, 2.5, 1.2, 0.05)
        reduction = st.slider("Capacity reduction (%)", 0, 80, 0, 5)
        run = st.form_submit_button("Compare network scenario")
    if run:
        try:
            result = post("/scenarios/network", {"demand_multiplier": multiplier, "capacity_reduction_pct": reduction})
            a, b, c = st.columns(3)
            a.metric("Centers over capacity", result["centers_over_capacity"])
            b.metric("Total excess / hr", f'{result["total_over_capacity_per_hour"]:,.0f}')
            c.metric("Highest risk centers", ", ".join(result["highest_risk_centers"]))
            st.dataframe(pd.DataFrame(result["centers"]), use_container_width=True, hide_index=True)
        except Exception as exc: st.error(f"Scenario comparison failed: {exc}")

with tab_history:
    st.subheader("Workflow executions")
    if workflows:
        st.dataframe(pd.DataFrame(workflows), use_container_width=True, hide_index=True)
        selected = st.selectbox("Inspect workflow", [w["id"] for w in workflows])
        detail = get(f"/workflows/{selected}", {})
        if detail:
            st.json(detail)
            if detail.get("status") == "FAILED" and st.button("Retry failed workflow"):
                try: st.success(post(f"/workflows/{selected}/retry", {})["status"])
                except Exception as exc: st.error(f"Retry failed: {exc}")
            st.subheader("Audit trail")
            st.dataframe(pd.DataFrame(get(f"/audit?entity_id={selected}", [])), use_container_width=True, hide_index=True)
    st.subheader("Recent events")
    if events: st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
    st.subheader("Recent audit records")
    audit = get("/audit?limit=50", [])
    if audit: st.dataframe(pd.DataFrame(audit), use_container_width=True, hide_index=True)

st.caption("Planning uses deterministic, explainable heuristics. Transfer recommendations require human approval; no real-world operations are executed.")
