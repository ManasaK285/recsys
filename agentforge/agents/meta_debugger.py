from __future__ import annotations
from agents.base import LLM
class MetaDebugger:
    def __init__(self, llm: LLM): self.llm = llm
    def diagnose(self, failures: list[dict], traces: list[dict]) -> dict:
        prompt = str({"failures": failures[-10:], "traces": traces[-10:]})
        result = self.llm.complete_json("You are the AgentForge meta-debugger. Diagnose workflow failures and propose harness interventions as JSON.", prompt)
        return result
