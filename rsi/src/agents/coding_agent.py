import os
from typing import Optional
from src.tasks.registry import Task

class CodingAgent:
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm and bool(os.getenv("OPENAI_API_KEY"))

    def generate(self, task: Task, approach: str, parent_code: Optional[str] = None) -> str:
        # The deterministic path makes the repository runnable without external services.
        factory = task.candidate_factories[approach]
        return factory()
