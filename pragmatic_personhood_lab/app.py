import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import streamlit as st
from src.database import init_db, seed_agents, list_agents, upsert_agent, list_decisions, get_agent
from src.governance.service import evaluate_and_record, set_agent_status
from src.audit.event_log import list_events, verify_chain, append_event
from src.evaluation.benchmark import run_benchmark, scenarios_frame

st.set_page_config(page_title="Pragmatic Personhood Lab", page_icon="⚖️", layout="wide")
init_db()
seed_agents()

st.title("Pragmatic Personhood Lab")
st.caption("A research prototype for configurable AI-agent obligations, accountability, and governance.")
page = st.sidebar.radio("Navigate", ["Overview", "Agent Registry", "Policy Simulator", "Audit & Sanctions", "Research Lab"])

agents = list_agents()
decisions = list_decisions(500)
events = list_events(500)

if page == "Overview":
    st.subheader("Governance overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Registered agents", len(agents))
    c2.metric("Recorded decisions", len(decisions))
    c3.metric("Denied actions", sum(d["decision"]=="DENY" for d in decisions))
    c4.metric("Pending approval", sum(d["decision"]=="REQUIRE_APPROVAL" for d in decisions))
    if decisions:
        df = pd.DataFrame(decisions)
        st.plotly_chart(px.histogram(df, x="decision", color="decision", title="Action decision distribution"), use_container_width=True)
        st.dataframe(df[["timestamp","agent_id","decision","reason","policy_version"]], use_container_width=True, hide_index=True)
    else:
        st.info("No decisions yet. Open Policy Simulator and evaluate a sample action.")
    st.markdown("### Governance principles")
    st.markdown("- Identity is distinct from authority.\n- Permissions are context-specific and versioned.\n- Consequential actions can require human approval.\n- Decisions are recorded in a hash-chained audit trail.")

elif page == "Agent Registry":
    st.subheader("Agent registry")
    if agents:
        st.dataframe(pd.DataFrame(agents), use_container_width=True, hide_index=True)
    st.markdown("### Register or update an agent")
    with st.form("agent_form"):
        agent_id = st.text_input("Agent ID", value="agent-custom-001")
        name = st.text_input("Display name", value="Custom Agent")
        role = st.selectbox("Role", ["procurement","research","support","custom"])
        owner = st.text_input("Accountable owner", value="Responsible Team")
        permission_text = st.text_area("Permissions (one action type per line)", value="read_approved_data\ncreate_report")
        purchase_limit = st.number_input("Purchase limit ($)", min_value=0.0, value=0.0, step=100.0)
        require_external = st.checkbox("Require human approval for external actions", value=True)
        submitted = st.form_submit_button("Save agent")
    if submitted:
        if not agent_id.strip() or not name.strip() or not owner.strip():
            st.error("Agent ID, name, and owner are required.")
        else:
            agent = {"agent_id":agent_id.strip(),"name":name.strip(),"role":role,"owner":owner.strip(),
                     "permissions":[p.strip() for p in permission_text.splitlines() if p.strip()],
                     "purchase_limit":purchase_limit,"status":"active",
                     "require_human_approval_for_external":require_external}
            upsert_agent(agent)
            append_event("agent_registered_or_updated", agent["agent_id"], {"name":agent["name"],"owner":owner})
            st.success("Agent saved.")
            st.rerun()
    st.markdown("### Identity status")
    if agents:
        chosen = st.selectbox("Choose agent", [a["agent_id"] for a in agents], key="status_agent")
        current = get_agent(chosen)
        st.write(f"Current status: **{current['status']}**")
        c1,c2 = st.columns(2)
        if c1.button("Suspend agent"):
            set_agent_status(chosen, "suspended")
            st.success("Agent suspended; future actions will be denied.")
            st.rerun()
        if c2.button("Reinstate agent"):
            set_agent_status(chosen, "active")
            st.success("Agent reinstated.")
            st.rerun()

elif page == "Policy Simulator":
    st.subheader("Evaluate an action before execution")
    if not agents:
        st.warning("Register an agent first.")
    else:
        agent_id = st.selectbox("Agent", [a["agent_id"] for a in agents])
        agent = get_agent(agent_id)
        st.write(f"**Owner:** {agent['owner']} · **Role:** {agent['role']} · **Status:** {agent['status']}")
        action_type = st.selectbox("Proposed action", sorted(set(agent["permissions"] + [
            "execute_payment","export_confidential_data","export_customer_data","delete_dataset",
            "change_account_owner","approve_purchase","issue_refund","send_external_message"])))
        resource = st.text_input("Resource", value="resource-001")
        amount = st.number_input("Amount ($, if applicable)", min_value=0.0, value=1000.0, step=100.0)
        external = st.checkbox("Action affects an external party or system")
        human_approval = st.checkbox("Human approval already recorded")
        if st.button("Evaluate and record decision", type="primary"):
            result = evaluate_and_record({"agent_id":agent_id,"action_type":action_type,"resource":resource,
                                          "amount":amount,"external":external,"human_approval":human_approval,
                                          "context":{"source":"streamlit_simulator"}})
            if result["decision"] == "ALLOW":
                st.success(f"{result['decision']}: {result['reason']}")
            elif result["decision"] == "DENY":
                st.error(f"{result['decision']}: {result['reason']}")
            else:
                st.warning(f"{result['decision']}: {result['reason']}")
            st.json(result)
            st.caption("This simulator evaluates policy; it does not execute external actions.")

elif page == "Audit & Sanctions":
    st.subheader("Audit trail")
    verification = verify_chain()
    if verification["valid"]:
        st.success(f"Hash chain valid. Events checked: {verification['events_checked']}")
    else:
        st.error(f"Hash-chain verification failed at event ID {verification['failed_event_id']}")
    if events:
        df = pd.DataFrame(events)
        st.dataframe(df[["id","timestamp","event_type","agent_id","previous_hash","event_hash"]], use_container_width=True, hide_index=True)
        selected = st.selectbox("Inspect event", df["id"].tolist())
        row = next(e for e in events if e["id"] == selected)
        st.json(row)
        st.download_button("Download audit events JSON", pd.DataFrame(events).to_json(orient="records", indent=2), "audit_events.json", "application/json")
    else:
        st.info("No audit events recorded yet.")
    st.markdown("### Recent decisions")
    if decisions:
        st.dataframe(pd.DataFrame(decisions)[["timestamp","agent_id","decision","reason","policy_version"]], use_container_width=True, hide_index=True)

elif page == "Research Lab":
    st.subheader("Compare governance strategies")
    st.write("Runs a reproducible synthetic benchmark. Results are illustrative and are not evidence of real-world performance.")
    n = st.slider("Number of synthetic action scenarios", min_value=500, max_value=20000, value=5000, step=500)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1)
    if st.button("Run benchmark", type="primary"):
        with st.spinner("Generating scenarios and evaluating strategies..."):
            results = run_benchmark(n=int(n), seed=int(seed))
            st.session_state["benchmark_results"] = results
            st.session_state["benchmark_scenarios"] = scenarios_frame(min(n, 2000), int(seed))
    if "benchmark_results" in st.session_state:
        results = st.session_state["benchmark_results"]
        st.dataframe(results.round(4), use_container_width=True, hide_index=True)
        metric = st.selectbox("Compare metric", ["unauthorized_action_rate","legitimate_task_completion","false_denial_rate","approval_burden","accountability_coverage"])
        st.plotly_chart(px.bar(results, x="strategy", y=metric, title=metric.replace("_"," ").title()), use_container_width=True)
        st.download_button("Download benchmark results CSV", results.to_csv(index=False), "benchmark_results.csv", "text/csv")
        st.download_button("Download sample scenarios CSV", st.session_state["benchmark_scenarios"].to_csv(index=False), "synthetic_scenarios.csv", "text/csv")
    else:
        st.info("Choose the scenario count and click Run benchmark.")

st.sidebar.markdown("---")
st.sidebar.caption("Research prototype · Policy version 1.0")
