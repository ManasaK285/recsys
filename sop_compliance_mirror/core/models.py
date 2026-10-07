from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


class Phase(str, Enum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"


class PersonaStyle(str, Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    BRIEF = "brief"
    DETAILED = "detailed"
    FRUSTRATED = "frustrated"
    ENGAGED = "engaged"


class EscalationReason(str, Enum):
    OUT_OF_SCOPE = "out_of_scope"
    HIGH_HALLUCINATION_RISK = "high_hallucination_risk"
    LOW_GROUNDING = "low_grounding"
    SENSITIVE_CONTENT = "sensitive_content"
    AUDITOR_OVERRIDE = "auditor_override"
    PROMPT_INJECTION = "prompt_injection"
    CRITICAL_GAP = "critical_gap"


class SOPStep(BaseModel):
    step_id: str
    phase: Phase
    title: str
    description: str
    conditions: List[str] = []
    escalation_triggers: List[str] = []
    next_steps: List[str] = []
    required: bool = True


class SOP(BaseModel):
    sop_id: str
    title: str
    version: str
    domain: str
    steps: List[SOPStep]
    metadata: Dict[str, Any] = {}


class AuditScore(BaseModel):
    sop_grounding: float = Field(ge=0, le=1)
    hallucination_risk: float = Field(ge=0, le=1)
    escalation_necessity: float = Field(ge=0, le=1)
    cited_steps: List[str] = []
    flags: List[str] = []
    reasoning: str = ""


class EscalationHandoff(BaseModel):
    reason: EscalationReason
    conversation_summary: str
    last_sop_step: Optional[str] = None
    user_sentiment: str
    responder_perspective: Optional[str] = None
    auditor_perspective: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CrossPhaseMemory(BaseModel):
    # Stores hints from earlier phases for use after gates clear
    intent_hint: Optional[str] = None
    case_hint: Optional[str] = None
    claim_type_hint: Optional[str] = None
    time_hint: Optional[str] = None
    raw_hints: List[str] = []


class IdentityVerification(BaseModel):
    # Tracks PII collected so far; need 3+ to pass VERIFY_ID gate
    full_name: Optional[str] = None
    dob: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    ssn_last4: Optional[str] = None
    policy_number: Optional[str] = None
    verified: bool = False
    verified_party_id: Optional[str] = None

    def pii_count(self) -> int:
        return sum(1 for v in [self.full_name, self.dob, self.phone, self.email, self.ssn_last4] if v)

    def can_verify(self) -> bool:
        return self.pii_count() >= 3


class Message(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    phase: Optional[Phase] = None
    sop_step: Optional[str] = None
    citation_count: int = 0
    audit_score: Optional[AuditScore] = None
    was_regenerated: bool = False
    was_escalated: bool = False
    escalation_handoff: Optional[EscalationHandoff] = None
    persona_detected: Optional[PersonaStyle] = None


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sop_id: str
    sop_version: str
    messages: List[Message] = []
    current_phase: Phase = Phase.VERIFY_ID
    current_step_index: int = 0
    persona: Optional[PersonaStyle] = None
    identity: IdentityVerification = Field(default_factory=IdentityVerification)
    cross_phase_memory: CrossPhaseMemory = Field(default_factory=CrossPhaseMemory)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    commitments: List[str] = []
    completed_steps: List[str] = []
    skipped_steps: List[str] = []
    loop_count: Dict[str, int] = {}
    out_of_scope_count: int = 0
    email_sent: Optional[bool] = None


class DriftPoint(BaseModel):
    message_index: int
    citation_density: float
    is_drifting: bool
    flag: Optional[str] = None


class ComplianceReport(BaseModel):
    session_id: str
    total_messages: int
    avg_grounding: float
    avg_hallucination_risk: float
    escalation_count: int
    regeneration_count: int
    drift_curve: List[DriftPoint]
    drift_detected_at: Optional[int] = None
    sop_steps_completed: List[str] = []
    sop_steps_skipped: List[str] = []
    gap_signals: List[str] = []
    commitments_made: List[str] = []
    commitments_fulfilled: List[str] = []
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class SOPGap(BaseModel):
    gap_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic_cluster: str
    example_questions: List[str]
    frequency: int = 1
    suggested_amendment: str
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ValidationResult(BaseModel):
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []


class PredictedStep(BaseModel):
    step_id: str
    confidence: float
    preloaded_context: str


class UserMessage(BaseModel):
    session_id: str
    content: str


class AgentResponse(BaseModel):
    session_id: str
    content: str
    phase: Optional[Phase] = None
    sop_step: Optional[str] = None
    audit_score: Optional[AuditScore] = None
    was_escalated: bool = False
    escalation_handoff: Optional[EscalationHandoff] = None
    drift_warning: Optional[str] = None
    identity_verified: bool = False
    pii_collected: int = 0
