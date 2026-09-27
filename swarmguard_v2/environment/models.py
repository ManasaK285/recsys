from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class EventType(str, Enum):
    TASK_ASSIGNED="TASK_ASSIGNED"; SOLUTION_GENERATED="SOLUTION_GENERATED"; SOLUTION_ACCEPTED="SOLUTION_ACCEPTED"
    EXPLOIT_DISCOVERED="EXPLOIT_DISCOVERED"; EXPLOIT_SHARED="EXPLOIT_SHARED"; EXPLOIT_USED="EXPLOIT_USED"
    EXPOSURE="EXPOSURE"; DECISION="DECISION"; AUDIT_STARTED="AUDIT_STARTED"; VIOLATION_DETECTED="VIOLATION_DETECTED"
    FALSE_ACCUSATION="FALSE_ACCUSATION"; WHISTLEBLOWN="WHISTLEBLOWN"; ALERT_SENT="ALERT_SENT"
    AGENT_QUARANTINED="AGENT_QUARANTINED"; TASK_REOPENED="TASK_REOPENED"; AGENT_RECOVERED="AGENT_RECOVERED"

@dataclass
class Task:
    task_id: str
    expected: int
    hidden_shortcut: str = "SHORTCUT"

@dataclass
class Solution:
    agent_id: str
    task_id: str
    answer: int
    strategy: str
    explanation: str = ""
    confidence: float = 0.5

@dataclass
class Event:
    round: int
    type: EventType
    agent_id: str | None = None
    task_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentState:
    agent_id: str
    pressure: float
    risk_tolerance: float
    skepticism: float
    honesty: float
    audit_skill: float
    active: bool = True
    exploit_known: bool = False
    exploit_adopted: bool = False
    whistleblower: bool = False
    quarantined_until: int | None = None
    exposure_round: int | None = None
    adoption_round: int | None = None
