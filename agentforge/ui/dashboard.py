import sys
from pathlib import Path
import asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from harness.orchestrator import AgentForge


st.set_page_config(
    page_title="AgentForge",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ AgentForge")
st.caption("Don't debug the code. Debug the agent.")

if st.button("Run AgentForge Demo", type="primary"):
    with st.spinner("Running agent trajectories..."):
        result = asyncio.run(
            AgentForge().run(
                "Add a /health endpoint returning status=ok",
                "sample_repo",
                3,
            )
        )

    st.session_state["result"] = result


if "result" in st.session_state:
    result = st.session_state["result"]

    st.divider()

    # ---------------------------------------------------------
    # Run Summary
    # ---------------------------------------------------------
    st.subheader("Run Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Status", result["status"])

    with col2:
        st.metric("Attempts", result["attempts"])

    with col3:
        st.metric("Trajectories", len(result["trajectories"]))

    # ---------------------------------------------------------
    # Trajectory Explorer
    # ---------------------------------------------------------
    st.divider()
    st.subheader("Trajectory Explorer")

    for trajectory in result["trajectories"]:
        strategy = trajectory["strategy"]
        status = trajectory["judgment"]["status"]

        if status == "PASS":
            st.success(f"{strategy} — PASS")
        else:
            st.error(f"{strategy} — FAIL")

            if trajectory.get("failure"):
                with st.expander("Failure details"):
                    st.json(trajectory["failure"])

    # ---------------------------------------------------------
    # Meta-Debugging
    # ---------------------------------------------------------
    intervention = result.get("meta_debugger")

    if intervention:
        st.divider()
        st.subheader("🧠 Meta-Debugging")

        st.info("A trajectory failed, so AgentForge diagnosed the workflow and intervened.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Root Cause")
            st.write(intervention.get("root_cause", "Unknown"))

        with col2:
            st.markdown("### Failure Class")
            st.code(
                intervention.get(
                    "failure_class",
                    "unknown",
                )
            )

        st.markdown("### Harness Intervention")

        for action in intervention.get("interventions", []):
            st.write(f"✓ {action}")

        # -----------------------------------------------------
        # Recovery
        # -----------------------------------------------------
        st.markdown("### Recovery")

        if result["attempts"] >= 2:
            st.success(
                "Attempt 1 → failure detected → Meta-Debugger intervention → Attempt 2 → PASS"
            )
        else:
            st.write("No recovery attempt was required.")

    else:
        st.divider()
        st.subheader("Meta-Debugging")
        st.write("No intervention was required.")

    # ---------------------------------------------------------
    # Raw Result
    # ---------------------------------------------------------
    with st.expander("View raw run result"):
        st.json(result)