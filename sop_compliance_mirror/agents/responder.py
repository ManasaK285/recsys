import re
import json
import anthropic
from typing import Optional, List, Tuple, Dict, Any
from core.models import (
    SOP, Session, Message, Phase, PersonaStyle,
    IdentityVerification, CrossPhaseMemory
)

MODEL = "claude-sonnet-4-6"

EMOTIONAL_SIGNALS = {
    "frustrated": ["ridiculous", "unacceptable", "terrible", "useless", "hate", "worst", "angry", "furious"],
    "anxious": ["worried", "scared", "nervous", "please help", "desperate", "urgent", "need this now"],
    "confused": ["don't understand", "what does that mean", "i'm lost", "confused", "unclear"],
    "refusing": ["won't give", "not giving", "refuse", "not telling you", "privacy", "why do you need"],
}

DE_ESCALATION_PROMPTS = {
    "frustrated": (
        "I completely understand how frustrating this must be. "
        "Your time matters, and I want to resolve this as quickly as possible for you. "
    ),
    "anxious": (
        "I hear you, and I want you to know we're going to work through this together. "
    ),
    "confused": (
        "Let me make this clearer for you. "
    ),
    "refusing": (
        "I completely respect your privacy concerns. "
        "The verification step exists to protect you, not to collect unnecessary data. "
        "We only need 3 pieces of information to confirm your identity. "
    ),
}


def detect_emotion(text: str) -> Optional[str]:
    text_lower = text.lower()
    for emotion, signals in EMOTIONAL_SIGNALS.items():
        if any(s in text_lower for s in signals):
            return emotion
    if text.count("!") > 2:
        return "frustrated"
    return None


def detect_persona(messages: List[Message]) -> Optional[PersonaStyle]:
    user_msgs = [m for m in messages if m.role == "user"][:3]
    if not user_msgs:
        return None
    all_text = " ".join(m.content for m in user_msgs)
    avg_len = sum(len(m.content.split()) for m in user_msgs) / len(user_msgs)
    emotion = detect_emotion(all_text)
    if emotion == "frustrated":
        return PersonaStyle.FRUSTRATED
    if avg_len < 8:
        return PersonaStyle.BRIEF
    if any(w in all_text.lower() for w in ["please", "kindly", "could you", "would you"]):
        return PersonaStyle.FORMAL
    if any(w in all_text.lower() for w in ["hey", "yeah", "ok", "cool"]):
        return PersonaStyle.CASUAL
    return PersonaStyle.ENGAGED


def extract_cross_phase_hints(text: str, memory: CrossPhaseMemory) -> CrossPhaseMemory:
    text_lower = text.lower()

    # Intent hints
    if any(w in text_lower for w in ["denied", "rejection", "rejected"]):
        memory.intent_hint = "denied_claim"
    elif any(w in text_lower for w in ["appeal", "fight", "dispute"]):
        memory.intent_hint = "appeal"
    elif any(w in text_lower for w in ["status", "update", "check"]):
        memory.intent_hint = "status_check"
    elif any(w in text_lower for w in ["submit", "upload", "send document"]):
        memory.intent_hint = "document_submission"

    # Claim type hints
    for ctype in ["healthcare", "health", "medical", "dental", "auto", "car"]:
        if ctype in text_lower:
            memory.claim_type_hint = "healthcare" if ctype in ["healthcare", "health", "medical"] else ctype
            break

    # Time hints
    time_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|\d{4}|last month|last year)\b', text_lower)
    if time_match:
        memory.time_hint = time_match.group()

    # Store raw hint
    hint = text[:200]
    if hint not in memory.raw_hints:
        memory.raw_hints.append(hint)

    return memory


def try_extract_pii(text: str, identity: IdentityVerification, fixtures: dict) -> IdentityVerification:
    text_lower = text.lower()

    # Full name: match against known policyholders
    for ph in fixtures["policyholders"].values():
        if ph["name"].lower() in text_lower:
            identity.full_name = ph["name"]
            break
        if "name_aliases" in ph:
            for alias in ph["name_aliases"]:
                if alias.lower() in text_lower:
                    identity.full_name = ph["name"]
                    break

    # DOB: various formats
    dob_match = re.search(r'\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-\d{1,2}-\d{4})\b', text)
    if dob_match:
        identity.dob = dob_match.group()

    # Phone: 10+ digit sequences
    phone_match = re.search(r'\+?1?\s*[\(\-\.]?\d{3}[\)\-\.\s]\s*\d{3}[\-\.\s]\d{4}', text)
    if phone_match:
        identity.phone = phone_match.group().strip()

    # Email
    email_match = re.search(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', text_lower)
    if email_match:
        identity.email = email_match.group()

    # SSN last 4: standalone 4-digit number not part of a date
    ssn_match = re.search(r'(?<![0-9/-])(\d{4})(?![0-9/-])', text)
    if ssn_match and not identity.ssn_last4:
        identity.ssn_last4 = ssn_match.group(1)

    # Policy number
    policy_match = re.search(r'\b(POL-\d+)\b', text, re.IGNORECASE)
    if policy_match:
        identity.policy_number = policy_match.group().upper()

    return identity


def try_verify_identity(identity: IdentityVerification, fixtures: dict) -> Optional[str]:
    if not identity.can_verify():
        return None

    for party_id, ph in fixtures["policyholders"].items():
        name_ok = identity.full_name and ph["name"].lower() == identity.full_name.lower()
        alias_ok = False
        if "name_aliases" in ph and identity.full_name:
            alias_ok = any(a.lower() == identity.full_name.lower() for a in ph["name_aliases"])

        if not (name_ok or alias_ok):
            continue

        # Check policy if provided
        if identity.policy_number and identity.policy_number != ph["policy_number"]:
            continue

        # Check at least one ID field
        dob_ok = identity.dob and identity.dob == ph["dob"]
        ssn_ok = identity.ssn_last4 and identity.ssn_last4 == ph["id_last4"]
        phone_ok = identity.phone and (
            re.sub(r'\D', '', identity.phone)[-10:] == re.sub(r'\D', '', ph.get("phone", ""))[-10:]
        )
        email_ok = identity.email and identity.email.lower() == ph.get("email", "").lower()

        if dob_ok or ssn_ok or phone_ok or email_ok:
            return party_id

    # Check representatives
    for rep in fixtures["representatives"]:
        if identity.full_name and rep["rep_name"].lower() == identity.full_name.lower():
            return rep["buyer_party_id"]

    return None


def build_system_prompt(
    sop: SOP,
    session: Session,
    fixtures: dict,
    emotion: Optional[str] = None,
) -> str:
    step = sop.steps[session.current_step_index]
    ph_data = ""
    if session.identity.verified and session.identity.verified_party_id:
        pid = session.identity.verified_party_id
        ph = fixtures["policyholders"].get(pid, {})
        claims = fixtures["claims_by_party"].get(pid, [])
        ph_data = f"POLICYHOLDER: {ph.get('name')} | Policy: {ph.get('policy_number')}\nCLAIMS:\n{json.dumps(claims, indent=2)}"

    mem = session.cross_phase_memory
    memory_block = ""
    if mem.intent_hint or mem.claim_type_hint or mem.time_hint:
        memory_block = (
            f"CROSS-PHASE MEMORY (use this, do not re-ask):\n"
            f"Intent: {mem.intent_hint}\nClaim type: {mem.claim_type_hint}\nTime: {mem.time_hint}\n"
        )

    emotion_block = ""
    if emotion and emotion in DE_ESCALATION_PROMPTS:
        emotion_block = (
            f"USER EMOTION: {emotion}\n"
            f"START with this de-escalation before anything else: '{DE_ESCALATION_PROMPTS[emotion]}'\n"
            f"Then continue with the SOP step naturally.\n"
        )

    persona_note = {
        PersonaStyle.FRUSTRATED: "Keep responses short and empathetic. No filler phrases.",
        PersonaStyle.BRIEF: "Be concise. Skip preamble.",
        PersonaStyle.FORMAL: "Use professional tone.",
        PersonaStyle.CASUAL: "Match casual tone.",
        PersonaStyle.ENGAGED: "Be thorough and helpful.",
    }.get(session.persona, "")

    pii_needed = []
    id = session.identity
    if not id.full_name: pii_needed.append("full name")
    if not id.dob: pii_needed.append("date of birth")
    if not any([id.phone, id.email, id.ssn_last4]):
        pii_needed.append("phone, email, or SSN last 4")
    pii_status = (
        f"PII collected: {id.pii_count()}/3 needed. "
        f"Still need: {', '.join(pii_needed) if pii_needed else 'GATE PASSED'}. "
        f"Verified: {id.verified}"
    )

    doc_guidance = json.dumps(sop.metadata.get("document_guidance", {}), indent=2)
    doc_alts = json.dumps(sop.metadata.get("document_alternative_guidance", {}), indent=2)

    out_of_scope_warning = ""
    if session.out_of_scope_count >= 2:
        out_of_scope_warning = "User has asked 2+ out-of-scope questions. On the next one, offer to transfer to a human representative."

    return f"""You are an insurance claims support agent following a strict SOP.

CURRENT PHASE: {step.phase.value} - {step.title}
PHASE INSTRUCTIONS: {step.description}

{pii_status}
{ph_data}
{memory_block}
{emotion_block}
PERSONA: {persona_note}
{out_of_scope_warning}

DOCUMENT GUIDANCE:
{doc_guidance}

DOCUMENT ALTERNATIVES:
{doc_alts}

CLAIM FOLLOWUP:
{json.dumps(sop.metadata.get('claim_followup_guidance', []), indent=2)}

RULES:
1. Always cite the phase as [PHASE_NAME] in your response
2. In VERIFY_ID: never reveal claim details before gate passes
3. In VERIFY_ID: if user provides intent hint, store it but stay in verification
4. In RESOLVE_INTENT: use cross-phase memory, do not re-ask what user already said
5. In POST_PROCESS: always offer email summary, user must choose send or skip
6. Out-of-scope questions: decline politely once; escalate if they repeat
7. Never fabricate amounts, deadlines, or document requirements
8. Keep responses under 280 words
9. If user is emotional, de-escalate first before continuing workflow
10. Explain SOP gates (like verification) when user pushes back, offer alternatives"""


def call_responder(
    client: anthropic.Anthropic,
    sop: SOP,
    session: Session,
    user_message: str,
    fixtures: dict,
    force_cite: bool = False
) -> Tuple[str, List[str], Optional[str]]:
    emotion = detect_emotion(user_message)

    if not session.persona and len(session.messages) >= 3:
        session.persona = detect_persona(session.messages)

    system = build_system_prompt(sop, session, fixtures, emotion)
    if force_cite:
        system += "\nYOU MUST cite [PHASE_NAME] for every claim. Previous response lacked citations."

    history = []
    for m in session.messages[-10:]:
        if m.role in ("user", "assistant"):
            history.append({"role": m.role, "content": m.content})
    history.append({"role": "user", "content": user_message})

    resp = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=system,
        messages=history
    )

    text = resp.content[0].text
    cited = re.findall(r'\[(VERIFY_ID|RESOLVE_INTENT|PROCESS_CASE|POST_PROCESS)\]', text)

    step = sop.steps[session.current_step_index]
    next_step_id = None
    if f"[{step.step_id}]" in text and step.step_id not in session.completed_steps:
        session.completed_steps.append(step.step_id)
        if session.current_step_index + 1 < len(sop.steps):
            next_step_id = sop.steps[session.current_step_index + 1].step_id

    return text, cited, next_step_id


def count_citations(text: str) -> int:
    return len(re.findall(r'\[(VERIFY_ID|RESOLVE_INTENT|PROCESS_CASE|POST_PROCESS)\]', text))
