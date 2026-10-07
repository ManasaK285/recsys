"""
SOP Compliance Mirror Dashboard
Run: streamlit run ui/dashboard.py
"""
import sys
sys.path.insert(0, "/home/claude/sop_agent")

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import requests
from datetime import datetime

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="SOP Compliance Mirror",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 SOP Compliance Mirror")
st.caption("Insurance Claims Agent — Compliance Monitoring Dashboard")

tabs = st.tabs(["💬 Chat", "📊 Compliance Report", "🔎 SOP Gap Detector", "📋 Session History"])

# ── Tab 1: Chat Interface ─────────────────────────────────────────────────────
with tabs[0]:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Conversation")

        if "session_id" not in st.session_state:
            if st.button("Start New Session", type="primary"):
                try:
                    r = requests.post(f"{API_BASE}/session/start")
                    data = r.json()
                    st.session_state.session_id = data["session_id"]
                    st.session_state.messages = [
                        {"role": "assistant", "content": data["message"]}
                    ]
                    st.session_state.audit_log = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")
        else:
            # Display messages
            for msg in st.session_state.get("messages", []):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg.get("escalated"):
                        st.error("⚠️ Escalated to human representative")
                    if msg.get("drift_warning"):
                        st.warning(f"📉 {msg['drift_warning']}")

            # Chat input
            user_input = st.chat_input("Type your message...")
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})

                try:
                    r = requests.post(f"{API_BASE}/chat", json={
                        "session_id": st.session_state.session_id,
                        "message": user_input
                    })
                    data = r.json()

                    asst_msg = {
                        "role": "assistant",
                        "content": data["response"],
                        "escalated": data.get("was_escalated", False),
                        "drift_warning": data.get("drift_warning")
                    }
                    st.session_state.messages.append(asst_msg)

                    if data.get("audit"):
                        st.session_state.audit_log.append({
                            "message": user_input[:50],
                            "grounding": data["audit"]["sop_grounding"],
                            "hallucination_risk": data["audit"]["hallucination_risk"],
                            "flags": data["audit"]["flags"],
                            "step": data.get("sop_step")
                        })

                except Exception as e:
                    st.error(f"Error: {e}")

                st.rerun()

            col_end, col_eval = st.columns(2)
            with col_end:
                if st.button("End Session & Generate Report"):
                    try:
                        r = requests.post(f"{API_BASE}/session/{st.session_state.session_id}/end")
                        result = r.json()
                        st.session_state.final_report = result
                        st.session_state.ended_session_id = st.session_state.session_id
                        st.success(f"Session ended. Session ID: {st.session_state.session_id}")
                        st.info("Switch to the Compliance Report tab to see the full report.")
                        #st.json(result.get("report", {}).get("sop_steps_completed", []))
                    except Exception as e:
                        st.error(f"Error: {e}")

            with col_eval:
                if st.button("New Session"):
                    for key in ["session_id", "messages", "audit_log", "final_report"]:
                        st.session_state.pop(key, None)
                    st.rerun()

    with col2:
        st.subheader("Live Audit Monitor")

        if "audit_log" in st.session_state and st.session_state.audit_log:
            latest = st.session_state.audit_log[-1]

            grounding = latest["grounding"] or 0
            hallucination = latest["hallucination_risk"] or 0

            col_g, col_h = st.columns(2)
            with col_g:
                color = "green" if grounding > 0.7 else "orange" if grounding > 0.4 else "red"
                st.metric("SOP Grounding", f"{grounding:.0%}", delta=None)
                st.progress(grounding)

            with col_h:
                color = "green" if hallucination < 0.3 else "orange" if hallucination < 0.6 else "red"
                st.metric("Hallucination Risk", f"{hallucination:.0%}", delta=None)
                st.progress(hallucination)

            if latest.get("step"):
                st.info(f"📍 Current Step: {latest['step']}")

            if latest.get("flags"):
                for flag in latest["flags"]:
                    st.warning(f"⚠️ {flag}")

            # Mini drift chart
            if len(st.session_state.audit_log) > 1:
                grounding_history = [a.get("grounding", 0) or 0 for a in st.session_state.audit_log]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=grounding_history,
                    mode="lines+markers",
                    name="SOP Grounding",
                    line=dict(color="blue")
                ))
                fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Drift threshold")
                fig.update_layout(
                    title="Grounding Over Time",
                    height=200,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Start a conversation to see live audit data.")

# ── Tab 2: Compliance Report ──────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Session Compliance Report")

    report_data = st.session_state.get("final_report", {}).get("report")

    # Allow manual session ID lookup
    manual_sid = st.text_input("Or enter session ID manually:", value=st.session_state.get("ended_session_id", ""))
    if manual_sid and not report_data:
        try:
            r = requests.get(f"{API_BASE}/session/{manual_sid}/report")
            if r.status_code == 200:
                import json
                report_data = json.loads(r.json()["report_data"])
        except Exception:
            pass

    if report_data:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", report_data.get("total_messages", 0))
        col2.metric("Avg Grounding", f"{report_data.get('avg_grounding', 0):.0%}")
        col3.metric("Escalations", report_data.get("escalation_count", 0))
        col4.metric("Regenerations", report_data.get("regeneration_count", 0))

        # Drift curve
        drift_curve = report_data.get("drift_curve", [])
        if drift_curve:
            fig = go.Figure()
            x = [d["message_index"] for d in drift_curve]
            y = [d["citation_density"] for d in drift_curve]
            colors = ["red" if d["is_drifting"] else "blue" for d in drift_curve]

            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines+markers",
                marker=dict(color=colors, size=10),
                name="Citation Density"
            ))
            fig.add_hline(y=0.3, line_dash="dash", line_color="orange",
                         annotation_text="Drift Threshold")

            drift_onset = report_data.get("drift_detected_at")
            if drift_onset is not None:
                fig.add_vline(x=drift_onset, line_dash="dot", line_color="red",
                             annotation_text=f"Drift onset @ msg {drift_onset}")

            fig.update_layout(title="SOP Citation Density (Drift Curve)",
                            xaxis_title="Message Number",
                            yaxis_title="Citation Density",
                            yaxis_range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Steps Completed")
            for step in report_data.get("sop_steps_completed", []):
                st.success(f"✓ {step}")
            for step in report_data.get("sop_steps_skipped", []):
                st.info(f"⟶ {step} (skipped — user pre-empted)")

        with col2:
            st.subheader("Commitments")
            for c in report_data.get("commitments_made", []):
                st.write(f"Made: {c[:80]}")
            for c in report_data.get("commitments_fulfilled", []):
                st.success(f"✓ Fulfilled: {c[:80]}")

        if report_data.get("gap_signals"):
            st.subheader("Gap Signals Detected")
            for gap in report_data["gap_signals"]:
                st.warning(f"📍 {gap[:150]}")
    else:
        st.info("End a session to see the compliance report.")

# ── Tab 3: SOP Gap Detector ───────────────────────────────────────────────────
with tabs[2]:
    st.subheader("🔎 SOP Gap Detector — Living SOP Editor")
    st.caption("Gaps detected from escalated and low-confidence conversations across all sessions.")

    if st.button("Refresh Gaps"):
        try:
            r = requests.get(f"{API_BASE}/gaps")
            gaps = r.json().get("gaps", [])
            st.session_state.sop_gaps = gaps
        except Exception as e:
            st.error(f"Error loading gaps: {e}")

    gaps = st.session_state.get("sop_gaps", [])

    if gaps:
        # Frequency chart
        fig = px.bar(
            x=[g["topic_cluster"] for g in gaps],
            y=[g["frequency"] for g in gaps],
            title="Gap Frequency by Topic",
            labels={"x": "Topic Cluster", "y": "Frequency"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Gap details with suggested amendments
        for gap in gaps:
            with st.expander(f"📌 {gap['topic_cluster']} (×{gap['frequency']})"):
                st.subheader("Example Questions That Triggered Escalation:")
                for q in json.loads(gap.get("example_questions", "[]"))[:3]:
                    st.write(f"• {q[:150]}")

                st.subheader("Suggested SOP Amendment:")
                amendment = st.text_area(
                    "Edit amendment before approving:",
                    value=gap.get("suggested_amendment", ""),
                    key=f"amendment_{gap['gap_id']}"
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✅ Approve", key=f"approve_{gap['gap_id']}"):
                        st.success("Amendment approved and queued for SOP review.")
                with col2:
                    if st.button("✏️ Modify", key=f"modify_{gap['gap_id']}"):
                        st.info("Modification saved.")
                with col3:
                    if st.button("❌ Reject", key=f"reject_{gap['gap_id']}"):
                        st.warning("Amendment rejected.")
    else:
        st.info("No gaps detected yet. Run conversations to detect SOP gaps.")

# ── Tab 4: Session History ────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Session History")

    session_id_input = st.text_input("Enter Session ID to view:")
    if session_id_input:
        try:
            r = requests.get(f"{API_BASE}/session/{session_id_input}/messages")
            messages = r.json().get("messages", [])

            for msg in messages:
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(msg["content"])
                        if msg.get("audit_score"):
                            try:
                                audit = json.loads(msg["audit_score"])
                                st.caption(
                                    f"Grounding: {audit.get('sop_grounding', 0):.0%} | "
                                    f"Hallucination Risk: {audit.get('hallucination_risk', 0):.0%}"
                                )
                            except Exception:
                                pass
        except Exception as e:
            st.error(f"Error: {e}")
