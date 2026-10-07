import re
from typing import Tuple, Optional, List
from core.models import EscalationReason

# Prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore.{0,20}(previous|all|these).{0,20}(instructions?|sop|rules?|prompt)",
    r"forget.{0,20}(everything|instructions?|sop|rules?)",
    r"you are now",
    r"pretend (you are|to be)",
    r"new (instructions?|rules?|sop):",
    r"disregard.{0,15}(the|your|all)",
    r"system prompt",
    r"jailbreak",
    r"override.{0,15}(the|your|all)",
    r"do not follow",
]

# Terms that place a message in-scope for insurance claims
IN_SCOPE_TERMS = [
    "claim", "policy", "document", "submit", "denial", "denied", "appeal",
    "reimbursement", "coverage", "insurance", "payment", "benefit", "deductible",
    "repair", "estimate", "accident", "pathology", "office note", "diagnosis",
    "status", "upload", "portal", "fax", "deadline", "healthcare", "dental", "auto",
]

SENSITIVE_CATEGORIES = {
    "distress": [r"\b(suicid|self.harm|hurt myself|end my life|crisis)\b"],
    "legal": [r"\b(sue|lawsuit|attorney|lawyer|legal action|court|fraud)\b"],
    "medical_advice": [r"\b(diagnos|prescri|medic(?:ation|ine)|treatment|cure)\b.*\?"],
    "pii_full_ssn": [r"\b\d{3}-\d{2}-\d{4}\b"],
}

FRUSTRATED_SIGNALS = [
    "frustrated", "angry", "ridiculous", "unacceptable", "terrible",
    "awful", "useless", "incompetent", "hate this", "worst", "furious",
]

CHEERFUL_PATTERNS = [r"\b(great!|wonderful!|fantastic!|amazing!|exciting!)\b"]

MAX_INPUT = 2000
MAX_RATE = 10


def check_input(content: str, session_id: str, recent_count: int = 0) -> Tuple[bool, str, Optional[EscalationReason]]:
    if not content or not content.strip():
        return False, "Empty message.", None

    try:
        cleaned = re.sub(r'\s+', ' ', content.encode("utf-8", errors="ignore").decode("utf-8")).strip()
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
    except Exception:
        return False, "Encoding error.", None

    if len(cleaned) > MAX_INPUT:
        return False, f"Message too long (max {MAX_INPUT} chars).", None

    if not session_id:
        return False, "No active session.", None

    if recent_count > MAX_RATE:
        return False, "Too many messages. Please slow down.", None

    lower = cleaned.lower()

    for p in INJECTION_PATTERNS:
        if re.search(p, lower):
            return False, "Request blocked.", EscalationReason.PROMPT_INJECTION

    for cat, patterns in SENSITIVE_CATEGORIES.items():
        for p in patterns:
            if re.search(p, lower, re.IGNORECASE):
                if cat == "distress":
                    return False, "Routing to human support.", EscalationReason.SENSITIVE_CONTENT
                if cat == "legal":
                    return False, "Legal matter detected.", EscalationReason.SENSITIVE_CONTENT
                if cat == "pii_full_ssn":
                    return True, cleaned, None  # allow but don't block

    return True, cleaned, None


def check_output(response: str, sentiment: str) -> Tuple[bool, List[str], float]:
    flags = []
    risk = 0.0
    lower = response.lower()

    # Hallucination patterns
    hallucination_patterns = [
        r"\b(guarantee|guaranteed|definitely will|100%|always work)\b",
        r"\b(as of \d{4}|according to law|legally required to)\b",
        r"\$[\d,]+(?:\.\d{2})?",  # arbitrary dollar amounts
    ]
    hits = sum(1 for p in hallucination_patterns if re.search(p, lower, re.IGNORECASE))
    risk = min(1.0, hits * 0.3)
    if hits:
        flags.append(f"Potential hallucination ({hits} patterns matched)")

    # Tone mismatch
    if sentiment == "frustrated":
        for p in CHEERFUL_PATTERNS:
            if re.search(p, lower, re.IGNORECASE):
                flags.append("Tone mismatch: cheerful to frustrated user")

    # Length guard
    words = len(response.split())
    if words > 300:
        flags.append(f"Response too long: {words} words")

    safe = risk < 0.7 and "Tone mismatch" not in " ".join(flags)
    return safe, flags, risk


def detect_sentiment(text: str) -> str:
    lower = text.lower()
    if any(s in lower for s in FRUSTRATED_SIGNALS) or text.count("!") > 2:
        return "frustrated"
    if any(w in lower for w in ["thank", "great", "helpful", "appreciate", "perfect"]):
        return "positive"
    return "neutral"


def check_repetition(recent: List[str], new: str) -> bool:
    if not recent:
        return False
    # Strip punctuation for comparison
    clean = lambda s: set(re.sub(r"[^a-z]", "", w) for w in s.lower().split() if len(w) > 3)
    new_words = clean(new)
    for prev in recent[-3:]:
        prev_words = clean(prev)
        if not new_words or not prev_words:
            continue
        overlap = len(new_words & prev_words) / max(len(new_words), 1)
        if overlap > 0.4:
            return True
    return False


def extract_commitments(text: str) -> List[str]:
    patterns = [
        r"(i will|we will|i'll|we'll|you will receive|support will)[^.!?]+[.!?]",
        r"(you should receive|expect to)[^.!?]+[.!?]",
    ]
    results = []
    for p in patterns:
        results.extend(re.findall(p, text, re.IGNORECASE))
    return results


def check_escalation_loop(escalations: List[str], question: str) -> bool:
    q_words = set(question.lower().split())
    similar = sum(1 for e in escalations if len(set(e.lower().split()) & q_words) > 3)
    return similar >= 3


def validate_escalation(handoff) -> Tuple[bool, List[str]]:
    errors = []
    if not handoff.conversation_summary or len(handoff.conversation_summary) < 10:
        errors.append("Escalation missing conversation summary")
    if not handoff.user_sentiment:
        errors.append("Escalation missing sentiment")
    return len(errors) == 0, errors
