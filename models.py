from dataclasses import dataclass, asdict
from typing import Optional, Any

@dataclass
class Evaluation:
    correct: bool
    runtime_ms: float
    memory_mb: float
    score: float
    tests_passed: int
    tests_total: int
    error: Optional[str] = None
    failure_class: Optional[str] = None

    def to_dict(self):
        return asdict(self)

@dataclass
class DiscoveryNode:
    node_id: str
    run_id: str
    task_id: str
    parent_id: Optional[str]
    depth: int
    approach: str
    code: str
    evaluation: Evaluation
    created_at: str
    policy_version: str

    def to_dict(self):
        d = asdict(self)
        d["evaluation"] = self.evaluation.to_dict()
        return d

@dataclass
class PolicyResult:
    policy_name: str
    reward: float
    work: int
    worlds: int
    details: Any
