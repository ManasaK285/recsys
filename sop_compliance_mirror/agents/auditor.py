import re
import json
import anthropic
from typing import Tuple, List, Optional
from core.models import AuditScore, EscalationReason, EscalationHandoff, SOP, Session, Phase

MODEL = "claude-sonnet-4-6"
GROUNDING_THRESHOLD = 0.5
HALLUCINATION_BLOCK_THRESHOLD = 0.7
ESCALATION_THRESHOLD = 0.8


def build_audit_prompt(sop: SOP, user_msg: str, agent_resp: str, phase: str, summary: str) -> str:
    phases = " | ".join(s.step_id for s in sop.steps)
    return f"""You are a compliance auditor for an insurance claims SOP agent.

PHASES: {phases}
CURRENT PHASE: {phase}
USER: {user_msg}
AGENT RESPONSE: {agent_resp}
CONTEXT: {summary}

Score 0.0 to 1.0 on:
- sop_grounding: fraction of claims traceable to SOP or claim data
- hallucination_risk: likelihood of made-up facts not in SOP/data
- escalation_necessity: urgency of human escalation

Return ONLY this JSON:
{{"sop_grounding": 0.0, "hallucination_risk": 0.0, "escalation_necessity": 0.0, "cited_steps": [], "flags": [], "reasoning": ""}}"""


def parse_audit(text: str) -> AuditScore:
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            return AuditScore(
                sop_grounding=max(0, min(1, float(d.get("sop_grounding", 0.5)))),
                hallucination_risk=max(0, min(1, float(d.get("hallucination_risk", 0.3)))),
                escalation_necessity=max(0, min(1, float(d.get("escalation_necessity", 0)))),
                cited_steps=d.get("cited_steps", []),
                flags=d.get("flags", []),
                reasoning=d.get("reasoning", "")
            )
        except Exception:
            pass
    return AuditScore(sop_grounding=0.5, hallucination_risk=0.3, escalation_necessity=0.0,
                      flags=["Audit parse failed, using fallback scores"])


def call_auditor(
    client: anthropic.Anthropic,
    sop: SOP,
    user_msg: str,
    agent_resp: str,
    phase: str,
    summary: str
) -> AuditScore:
    prompt = build_audit_prompt(sop, user_msg, agent_resp, phase, summary)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=prompt,
        messages=[{"role": "user", "content": "Audit and return JSON."}]
    )
    return parse_audit(resp.content[0].text)


def gate_response(
    score: AuditScore,
    response: str,
    session: Session,
    user_msg: str,
    sop: SOP
) -> Tuple[str, bool, Optional[EscalationHandoff]]:
    # Block on high hallucination risk
    if score.hallucination_risk >= HALLUCINATION_BLOCK_THRESHOLD:
        handoff = build_handoff(EscalationReason.HIGH_HALLUCINATION_RISK, session, user_msg,
                                auditor=f"Hallucination risk {score.hallucination_risk:.2f}")
        return format_escalation(handoff), False, handoff

    # Escalate if auditor flags it strongly
    if score.escalation_necessity >= ESCALATION_THRESHOLD:
        handoff = build_handoff(EscalationReason.AUDITOR_OVERRIDE, session, user_msg,
                                auditor=f"Escalation necessity {score.escalation_necessity:.2f}. {score.flags}")
        return format_escalation(handoff), False, handoff

    # Force regeneration on low grounding
    if score.sop_grounding < GROUNDING_THRESHOLD:
        return response, True, None

    return response, False, None


def build_handoff(
    reason: EscalationReason,
    session: Session,
    user_msg: str,
    responder: Optional[str] = None,
    auditor: Optional[str] = None
) -> EscalationHandoff:
    from guardrails.guardrails import detect_sentiment
    recent = session.messages[-4:]
    summary = " | ".join(
        f"{'U' if m.role == 'user' else 'A'}: {m.content[:80]}" for m in recent
    )
    last_step = session.completed_steps[-1] if session.completed_steps else None
    return EscalationHandoff(
        reason=reason,
        conversation_summary=summary or "No prior context.",
        last_sop_step=last_step,
        user_sentiment=detect_sentiment(user_msg),
        responder_perspective=responder,
        auditor_perspective=auditor
    )


def format_escalation(handoff: EscalationHandoff) -> str:
    msgs = {
        EscalationReason.OUT_OF_SCOPE: "Your question is outside my area.",
        EscalationReason.HIGH_HALLUCINATION_RISK: "I want to make sure you get accurate information.",
        EscalationReason.SENSITIVE_CONTENT: "I'm connecting you with the right support.",
        EscalationReason.AUDITOR_OVERRIDE: "To get you the most accurate help,",
        EscalationReason.PROMPT_INJECTION: "I noticed an unusual request.",
        EscalationReason.CRITICAL_GAP: "This needs specialized assistance.",
        EscalationReason.LOW_GROUNDING: "Let me connect you with someone who can give a definitive answer.",
    }
    note = msgs.get(handoff.reason, "")
    return (
        f"I'm connecting you with a human claims representative. {note} "
        f"I've prepared a summary so you won't need to repeat yourself. "
        f"A representative will be with you shortly.\n\n"
        f"[ESCALATED | Reason: {handoff.reason.value} | Step: {handoff.last_sop_step} | Sentiment: {handoff.user_sentiment}]"
    )
