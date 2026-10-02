from __future__ import annotations
from pydantic import BaseModel, Field
from agents.base import LLM

class TaskSpec(BaseModel):
    goal: str
    requirements: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=lambda: ["pytest", "ruff", "mypy"])
    relevant_files: list[str] = Field(default_factory=list)

class Planner:
    def __init__(self, llm: LLM): self.llm = llm
    def plan(self, task: str) -> TaskSpec:
        data = self.llm.complete_json("You are the AgentForge planning agent. Return JSON plan.", task)
        return TaskSpec(**data)
