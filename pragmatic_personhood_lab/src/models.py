from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class Agent:
    agent_id: str
    name: str
    role: str
    owner: str
    permissions: list[str]
    purchase_limit: float = 0.0
    status: str = "active"
    require_human_approval_for_external: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class ActionRequest:
    agent_id: str
    action_type: str
    resource: str = "unspecified"
    amount: float = 0.0
    external: bool = False
    human_approval: bool = False
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["context"] = self.context or {}
        return result
