import os
import anthropic
from typing import Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

from core.models import (
    Session, Message, AgentResponse, Phase,
    EscalationReason, EscalationHandoff
)
from core.sop_loader import load_sop, validate_sop, load_fixtures
from core.database import (
    init_db, save_session, save_message,
    save_compliance_report, save_sop_gap
)
from validation.validator import (
    validate_user_input, validate_agent_output, validate_session_state
)
from guardrails.guardrails import (
    check_input, check_output, check_repetition,
    extract_commitments, detect_sentiment, check_escalation_loop
)
from agents.responder import (
    call_responder, try_extract_pii, try_verify_identity,
    extract_cross_phase_hints, count_citations
)
from agents.auditor import (
    call_auditor, gate_response, build_handoff, format_escalation
)
from tracking.drift import generate_compliance_report, generate_sop_gaps

load_dotenv()

_sop = None
_fixtures = None
_client = None
_sessions: Dict[str, Session] = {}
_rate_counts: Dict[str, int] = {}
_escalations: Dict[str, list] = {}
_recent_responses: Dict[str, list] = {}


def initialize():
    global _sop, _fixtures, _client

    init_db()

    _sop = load_sop()
    result = validate_sop(_sop)
    if not result.valid:
        raise ValueError(f"SOP invalid: {result.errors}")

    _fixtures = load_fixtures()
    _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    return {"status": "ready", "sop": _sop.title, "version": _sop.version}


def create_session() -> str:
    s = Session(sop_id=_sop.sop_id, sop_version=_sop.version)
    _sessions[s.session_id] = s
    _recent_responses[s.session_id] = []
    _escalations[s.session_id] = []
    _rate_counts[s.session_id] = 0
    save_session(s)
    return s.session_id


def _advance_phase(session: Session):
    idx = session.current_step_index
    if idx + 1 < len(_sop.steps):
        session.current_step_index = idx + 1
        session.current_phase = _sop.steps[session.current_step_index].phase


def process_message(session_id: str, content: str) -> AgentResponse:
    if session_id not in _sessions:
        raise ValueError(f"Session {session_id} not found")

    session = _sessions[session_id]
    rate = _rate_counts.get(session_id, 0)

    # Step 1: validate input
    valid, cleaned, err = validate_user_input(content, session_id, rate)
    if not valid:
        return AgentResponse(session_id=session_id, content=f"Could not process: {err}")
    _rate_counts[session_id] = rate + 1

    # Step 2: input guardrails
    safe, cleaned, esc_reason = check_input(cleaned, session_id, rate)
    user_msg = Message(role="user", content=cleaned, phase=session.current_phase)
    session.messages.append(user_msg)
    save_message(user_msg, session_id)

    if not safe:
        handoff = build_handoff(esc_reason, session, cleaned)
        msg = format_escalation(handoff)
        asst = Message(role="assistant", content=msg, was_escalated=True, escalation_handoff=handoff)
        session.messages.append(asst)
        save_message(asst, session_id)
        save_session(session)
        return AgentResponse(session_id=session_id, content=msg, was_escalated=True, escalation_handoff=handoff)

    # Step 3: out-of-scope counter check
    from guardrails.guardrails import IN_SCOPE_TERMS
    in_scope = any(t in cleaned.lower() for t in IN_SCOPE_TERMS)
    if not in_scope and len(cleaned.split()) > 10:
        session.out_of_scope_count += 1

    # Step 4: extract PII and cross-phase memory in VERIFY_ID phase
    if session.current_phase == Phase.VERIFY_ID:
        session.identity = try_extract_pii(cleaned, session.identity, _fixtures)
        session.cross_phase_memory = extract_cross_phase_hints(cleaned, session.cross_phase_memory)

        # Try to verify with what we have
        if not session.identity.verified and session.identity.can_verify():
            party_id = try_verify_identity(session.identity, _fixtures)
            if party_id:
                session.identity.verified = True
                session.identity.verified_party_id = party_id
    else:
        # Store hints from any phase
        session.cross_phase_memory = extract_cross_phase_hints(cleaned, session.cross_phase_memory)

    # Step 5: detect persona
    if not session.persona and len(session.messages) >= 3:
        from agents.responder import detect_persona
        session.persona = detect_persona(session.messages)

    # Step 6: call responder
    step = _sop.steps[session.current_step_index]
    text, cited, next_step = call_responder(_client, _sop, session, cleaned, _fixtures)

    # Step 7: output validation
    step_ids = [s.step_id for s in _sop.steps]
    _, validated, _ = validate_agent_output(text, cleaned, step_ids, cited)

    # Step 8: output guardrails
    sentiment = detect_sentiment(cleaned)
    _, flags, h_risk = check_output(validated, sentiment)

    # Step 9: compliance auditor
    recent_ctx = " | ".join(
        f"{'U' if m.role == 'user' else 'A'}: {m.content[:60]}"
        for m in session.messages[-5:]
    )
    audit = call_auditor(_client, _sop, cleaned, validated, step.step_id, recent_ctx)
    audit.hallucination_risk = max(audit.hallucination_risk, h_risk)
    audit.flags.extend(flags)

    # Step 10: gate the response
    final, regen, handoff = gate_response(audit, validated, session, cleaned, _sop)
    regenerated = False

    if regen and not handoff:
        final, cited, next_step = call_responder(_client, _sop, session, cleaned, _fixtures, force_cite=True)
        regenerated = True
        audit = call_auditor(_client, _sop, cleaned, final, step.step_id, recent_ctx)

    # Step 11: session guardrails
    recent_r = _recent_responses.get(session_id, [])
    if check_repetition(recent_r, final):
        final += "\n\nIs there anything specific you'd like me to clarify?"
    recent_r.append(final)
    _recent_responses[session_id] = recent_r[-5:]

    new_commitments = extract_commitments(final)
    session.commitments.extend(new_commitments)

    if handoff:
        _escalations[session_id].append(cleaned)

    if check_escalation_loop(_escalations.get(session_id, []), cleaned):
        for g in generate_sop_gaps([cleaned], _sop.version):
            save_sop_gap(g)

    # Step 12: advance phase if verified or step completed
    if session.current_phase == Phase.VERIFY_ID and session.identity.verified:
        if session.current_step_index == 0:
            _advance_phase(session)
    elif next_step and not handoff:
        idx = next((i for i, s in enumerate(_sop.steps) if s.step_id == next_step), None)
        if idx is not None:
            session.current_step_index = idx
            session.current_phase = _sop.steps[idx].phase

    # Step 13: detect drift
    from tracking.drift import compute_citation_density, detect_drift_onset
    drift_warn = None
    if len([m for m in session.messages if m.role == "assistant"]) > 3:
        pts = compute_citation_density(session.messages)
        onset = detect_drift_onset(pts)
        if onset is not None:
            drift_warn = f"Drift detected from message {onset}"

    # Step 14: persist
    asst = Message(
        role="assistant",
        content=final,
        phase=session.current_phase,
        sop_step=step.step_id,
        citation_count=count_citations(final),
        audit_score=audit,
        was_regenerated=regenerated,
        was_escalated=handoff is not None,
        escalation_handoff=handoff,
        persona_detected=session.persona
    )
    session.messages.append(asst)
    if validate_session_state(session).valid:
        save_message(asst, session_id)
        save_session(session)

    return AgentResponse(
        session_id=session_id,
        content=final,
        phase=session.current_phase,
        sop_step=step.step_id,
        audit_score=audit,
        was_escalated=handoff is not None,
        escalation_handoff=handoff,
        drift_warning=drift_warn,
        identity_verified=session.identity.verified,
        pii_collected=session.identity.pii_count()
    )


def end_session(session_id: str) -> dict:
    if session_id not in _sessions:
        return {"error": "Session not found"}

    session = _sessions[session_id]
    session.ended_at = datetime.utcnow()

    report = generate_compliance_report(session)
    from validation.validator import validate_compliance_report
    if validate_compliance_report(report).valid:
        save_compliance_report(report)

    escalated = _escalations.get(session_id, [])
    if escalated:
        for g in generate_sop_gaps(escalated, _sop.version):
            save_sop_gap(g)

    save_session(session)
    return {"session_id": session_id, "report": report.model_dump(), "gaps_detected": len(escalated)}
