import uuid
import re
from typing import List, Optional, Dict
from core.models import DriftPoint, ComplianceReport, Session, SOPGap

DRIFT_THRESHOLD = 0.3


def compute_citation_density(messages: List) -> List[DriftPoint]:
    points = []
    asst_msgs = [m for m in messages if m.role == "assistant"]
    for i, msg in enumerate(asst_msgs):
        density = min(1.0, msg.citation_count / max(1, len(msg.content.split()) / 50))
        drifting = density < DRIFT_THRESHOLD and i > 2
        flag = None
        if drifting and points and points[-1].citation_density >= DRIFT_THRESHOLD:
            flag = f"Drift onset at message {i}"
        points.append(DriftPoint(message_index=i, citation_density=density, is_drifting=drifting, flag=flag))
    return points


def detect_drift_onset(points: List[DriftPoint]) -> Optional[int]:
    streak = 0
    for p in points:
        if p.is_drifting:
            streak += 1
            if streak >= 2:
                return p.message_index - 1
        else:
            streak = 0
    return None


def generate_compliance_report(session: Session) -> ComplianceReport:
    from guardrails.guardrails import check_repetition
    from validation.validator import validate_compliance_report

    asst = [m for m in session.messages if m.role == "assistant"]
    user = [m for m in session.messages if m.role == "user"]

    grounding = [m.audit_score.sop_grounding for m in asst if m.audit_score and not m.was_escalated]
    halluc = [m.audit_score.hallucination_risk for m in asst if m.audit_score and not m.was_escalated]

    avg_g = sum(grounding) / len(grounding) if grounding else 0.0
    avg_h = sum(halluc) / len(halluc) if halluc else 0.0

    drift_curve = compute_citation_density(session.messages)
    drift_onset = detect_drift_onset(drift_curve)

    escalations = sum(1 for m in asst if m.was_escalated)
    regen = sum(1 for m in asst if m.was_regenerated)

    gaps = []
    for i, um in enumerate(user):
        if i < len(asst):
            am = asst[i]
            if am.was_escalated or (am.audit_score and am.audit_score.sop_grounding < 0.4):
                gaps.append(um.content[:150])

    # Simple commitment fulfillment check
    all_resp_text = " ".join(m.content.lower() for m in asst)
    fulfilled = []
    unfulfilled = []
    for c in session.commitments:
        words = [w for w in c.lower().split() if len(w) > 4]
        if sum(1 for w in words if w in all_resp_text) >= len(words) * 0.5:
            fulfilled.append(c)
        else:
            unfulfilled.append(c)

    report = ComplianceReport(
        session_id=session.session_id,
        total_messages=len(session.messages),
        avg_grounding=round(avg_g, 3),
        avg_hallucination_risk=round(avg_h, 3),
        escalation_count=escalations,
        regeneration_count=regen,
        drift_curve=drift_curve,
        drift_detected_at=drift_onset,
        sop_steps_completed=session.completed_steps,
        sop_steps_skipped=session.skipped_steps,
        gap_signals=gaps,
        commitments_made=session.commitments,
        commitments_fulfilled=fulfilled
    )
    return report


def cluster_gaps(questions: List[str]) -> Dict[str, List[str]]:
    buckets = {
        "document_submission": ["submit", "upload", "send", "portal", "fax"],
        "appeal_process": ["appeal", "dispute", "challenge", "reconsider"],
        "timeline": ["when", "how long", "deadline", "days", "week"],
        "document_requirements": ["what documents", "what do i need", "which files"],
        "payment": ["payment", "reimbursement", "money", "paid", "amount"],
        "status": ["status", "update", "progress", "check", "track"],
        "alternatives": ["don't have", "can't get", "alternative", "substitute"],
    }
    clusters: Dict[str, List[str]] = {}
    for q in questions:
        ql = q.lower()
        matched = False
        for name, terms in buckets.items():
            if any(t in ql for t in terms):
                clusters.setdefault(name, []).append(q)
                matched = True
                break
        if not matched:
            clusters.setdefault("other", []).append(q)
    return clusters


def generate_sop_gaps(escalated: List[str], version: str) -> List[SOPGap]:
    if not escalated:
        return []
    clusters = cluster_gaps(escalated)
    amendments = {
        "document_submission": "Add step-by-step portal upload guide with accepted file types and sizes.",
        "appeal_process": "Add formal appeals section covering grounds, forms, timeline, and escalation path.",
        "timeline": "Add specific business day counts per claim type and proactive update schedule.",
        "document_requirements": "Add per-document checklists with format requirements and examples.",
        "payment": "Clarify allowed_max_amount vs expected_reimbursement vs net_pay with timeline after approval.",
        "status": "Define each status value and what triggers status changes.",
        "alternatives": "Expand alternative document guidance with ranked options and acceptance criteria.",
        "other": "Review uncategorized escalations for new SOP sections.",
    }
    return sorted([
        SOPGap(
            gap_id=str(uuid.uuid4()),
            topic_cluster=name,
            example_questions=qs[:5],
            frequency=len(qs),
            suggested_amendment=amendments.get(name, "Review and expand this section.")
        )
        for name, qs in clusters.items()
    ], key=lambda g: g.frequency, reverse=True)
