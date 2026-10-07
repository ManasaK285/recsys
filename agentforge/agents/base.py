from __future__ import annotations
import json, os
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLM:
    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        raise NotImplementedError


class MockLLM(LLM):
    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        if "plan" in system.lower():
            return {"goal": user[:200], "requirements": ["implement requested behavior", "preserve existing behavior", "add or update tests"], "verification": ["pytest", "ruff", "mypy"]}
        if "meta-debug" in system.lower():
            return {"root_cause": "workflow did not inspect tests and edge cases before implementation", "failure_class": "workflow.bad_recovery_strategy", "interventions": ["inspect existing tests before editing", "enumerate edge cases", "run the complete test suite before declaring success"]}
        return {"message": "mock response", "code_plan": user[:500]}


class OpenAICompatibleLLM(LLM):
    def __init__(self) -> None:
        if OpenAI is None:
            raise RuntimeError("Install the openai package")
        base_url = os.getenv("AGENTFORGE_LLM_BASE_URL")
        api_key = os.getenv("AGENTFORGE_LLM_API_KEY")
        model = os.getenv("AGENTFORGE_LLM_MODEL")
        if not (base_url and api_key and model):
            raise RuntimeError("LLM environment is incomplete")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return json.loads(response.choices[0].message.content or "{}")


def get_llm() -> LLM:
    if os.getenv("AGENTFORGE_LLM_BASE_URL") and os.getenv("AGENTFORGE_LLM_API_KEY") and os.getenv("AGENTFORGE_LLM_MODEL"):
        return OpenAICompatibleLLM()
    return MockLLM()
