import json
import os
import requests

class OpenAICompatibleOracle:
    def __init__(self, model=None, base_url=None, api_key=None):
        self.model = model or os.getenv("POLCA_MODEL", "gpt-4.1-mini")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM mode.")

    def propose(self, parent: str, feedback: str, summary: str, n: int = 2):
        prompt = f"""You are an optimizer for a customer-support response policy. Improve the policy while preserving useful behavior.

Current policy:\n{parent}

Latest evaluation feedback:\n{feedback}

Global optimization history:\n{summary}

Return exactly {n} distinct improved policies as a JSON array of strings. Do not explain them outside the JSON."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You propose concise, testable policy programs."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.8,
        }
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=90,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("["), text.rfind("]")
            data = json.loads(text[start:end + 1])
        return [str(x) for x in data][:n]
