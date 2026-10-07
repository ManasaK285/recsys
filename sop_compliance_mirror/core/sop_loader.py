import json
from pathlib import Path
from core.models import SOP, SOPStep, Phase, ValidationResult

FIXTURES = Path("data/fixtures")


def load_sop() -> SOP:
    g = json.loads((FIXTURES / "required_document_guideline.json").read_text())

    steps = [
        SOPStep(
            step_id="VERIFY_ID",
            phase=Phase.VERIFY_ID,
            title="Identity Verification",
            description=(
                "Collect at least 3 PII fields from: full name, DOB, phone, email, SSN last 4. "
                "Policy number helps with lookup but does not count as PII for the gate. "
                "Do NOT reveal claim details until gate passes. "
                "Store any intent or case hints the user mentions for use after verification. "
                "Accept authorized representatives on file."
            ),
            conditions=["need name + DOB + at least one of phone/email/ssn_last4"],
            escalation_triggers=["3 failed attempts", "user refuses all PII"],
            next_steps=["RESOLVE_INTENT"],
            required=True
        ),
        SOPStep(
            step_id="RESOLVE_INTENT",
            phase=Phase.RESOLVE_INTENT,
            title="Intent Resolution",
            description=(
                "Identify what the caller needs. Use cross-phase memory to skip re-asking "
                "if intent was stated during VERIFY_ID. "
                "Match to: denied claim appeal, document submission, status check, general inquiry. "
                "Resolve ambiguous language naturally."
            ),
            conditions=["identity verified in VERIFY_ID"],
            escalation_triggers=["intent cannot be determined after 2 attempts"],
            next_steps=["PROCESS_CASE"],
            required=True
        ),
        SOPStep(
            step_id="PROCESS_CASE",
            phase=Phase.PROCESS_CASE,
            title="Case Processing",
            description=(
                "Look up the caller's claims. Explain denial reason and documents needed. "
                "Provide document-specific guidance from guidelines. "
                "Offer submission methods (portal, fax, mail). "
                "State appeal deadline. "
                "Answer follow-up questions grounded only in claim data and SOP. "
                + g["default_guidance"]["en"]
            ),
            conditions=["intent resolved in RESOLVE_INTENT"],
            escalation_triggers=["legal threat", "all alternatives exhausted", "caller requests human"],
            next_steps=["POST_PROCESS"],
            required=True
        ),
        SOPStep(
            step_id="POST_PROCESS",
            phase=Phase.POST_PROCESS,
            title="Post-Processing and Email Summary",
            description=(
                "Offer to send the caller an email summary of the conversation, "
                "including what was discussed, claim status/outcome, and next steps. "
                "The caller must choose: send email or skip. "
                "If sending, confirm their email address. "
                "Confirm all questions are answered. "
                "Close the conversation politely."
            ),
            conditions=["case processed in PROCESS_CASE"],
            escalation_triggers=["unresolved issues remain"],
            next_steps=[],
            required=True
        ),
    ]

    return SOP(
        sop_id="insurance_claims_v1",
        title="Insurance Claims SOP",
        version="1.0.0",
        domain="insurance_claims",
        steps=steps,
        metadata={
            "claim_followup_guidance": g["claim_followup_guidance"],
            "claim_followup_fallback": g["claim_followup_fallback"]["en"],
            "document_guidance": g["document_guidance"],
            "document_alternative_guidance": g["document_alternative_guidance"],
            "case_type_guidance": g["case_type_guidance"],
            "average_processing_time": g["claim_followup_settings"]["average_processing_time_after_submission"]["en"]
        }
    )


def validate_sop(sop: SOP) -> ValidationResult:
    errors, warnings = [], []

    if not sop.steps:
        errors.append("SOP has no steps")

    all_ids = {s.step_id for s in sop.steps}
    for step in sop.steps:
        for nxt in step.next_steps:
            if nxt not in all_ids:
                errors.append(f"{step.step_id} references unknown step: {nxt}")

    ambiguous = ["usually", "generally", "it depends", "sometimes"]
    for step in sop.steps:
        for word in ambiguous:
            if word in step.description.lower():
                warnings.append(f"{step.step_id} has ambiguous language: '{word}'")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def load_fixtures() -> dict:
    claims = json.loads((FIXTURES / "claims.json").read_text())
    policyholders = json.loads((FIXTURES / "policyholders.json").read_text())
    representatives = json.loads((FIXTURES / "representatives.json").read_text())
    guidelines = json.loads((FIXTURES / "required_document_guideline.json").read_text())

    return {
        "claims": {c["case_id"]: c for c in claims},
        "policyholders": {p["party_id"]: p for p in policyholders},
        "policyholders_by_policy": {p["policy_number"]: p for p in policyholders},
        "representatives": representatives,
        "guidelines": guidelines,
        # index claims by party_id for fast lookup
        "claims_by_party": {
            pid: [c for c in claims if c["party_id"] == pid]
            for pid in set(c["party_id"] for c in claims)
        }
    }
