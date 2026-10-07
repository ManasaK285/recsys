import re
from typing import Tuple, List
from core.models import ValidationResult, Message, ComplianceReport, Session

MAX_INPUT = 2000
MAX_RATE = 10


def validate_user_input(content: str, session_id: str, recent_count: int = 0) -> Tuple[bool, str, str]:
    if not content or not content.strip():
        return False, "", "Empty message."
    try:
        cleaned = re.sub(r'\s+', ' ', content.encode("utf-8", errors="ignore").decode()).strip()
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
    except Exception:
        return False, "", "Encoding error."
    if len(cleaned) > MAX_INPUT:
        return False, "", f"Message too long (max {MAX_INPUT})."
    if not session_id:
        return False, "", "No active session."
    if recent_count > MAX_RATE:
        return False, "", "Rate limit exceeded."
    return True, cleaned, ""


def validate_agent_output(
    response: str, user_msg: str, sop_ids: list, cited: list
) -> Tuple[bool, str, List[str]]:
    errors = []
    if not response or not response.strip():
        errors.append("Empty agent response.")
        return False, "", errors

    # Enforce length budget
    if len(response) > 1500:
        cut = response[:1500]
        last_period = cut.rfind('.')
        response = (cut[:last_period + 1] if last_period > 1000 else cut) + " [Truncated.]"

    # Validate cited step IDs exist
    for step in cited:
        clean = step.strip("[]")
        if clean not in sop_ids:
            errors.append(f"Cited non-existent step: {step}")

    return len(errors) == 0, response, errors


def validate_session_state(session: Session) -> ValidationResult:
    errors, warnings = [], []
    if not session.session_id:
        errors.append("Missing session_id")
    if session.current_step_index < 0:
        errors.append("Negative step index")
    for k, v in session.loop_count.items():
        if v < 0:
            errors.append(f"Negative loop count: {k}")
            session.loop_count[k] = 0
    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_compliance_report(report: ComplianceReport) -> ValidationResult:
    errors, warnings = [], []
    for field in ["session_id", "total_messages", "avg_grounding", "avg_hallucination_risk"]:
        if getattr(report, field, None) is None:
            errors.append(f"Report missing: {field}")
    if not (0 <= report.avg_grounding <= 1):
        errors.append(f"avg_grounding out of range: {report.avg_grounding}")
        report.avg_grounding = max(0, min(1, report.avg_grounding))
    if not (0 <= report.avg_hallucination_risk <= 1):
        errors.append(f"avg_hallucination_risk out of range")
        report.avg_hallucination_risk = max(0, min(1, report.avg_hallucination_risk))
    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
