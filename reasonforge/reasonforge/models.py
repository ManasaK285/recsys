from __future__ import annotations

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class Factor(BaseModel):
    name: str
    values: List[str]


class Taxonomy(BaseModel):
    domain: str
    task: str
    factors: List[Factor]


class Scenario(BaseModel):
    sample_id: str
    intent: str
    factors: Dict[str, str]
    complexity: str
    instruction: str
    expected_label: str
    answer: str = ""
    critique_1: str = ""
    critique_2: str = ""
    quality_score: float = 0.0
    accepted: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
