from typing import List
from .candidate import Candidate

class LocalSummarizer:
    """Compact global context without requiring an external LLM."""
    def summarize(self, candidates: List[Candidate], limit: int = 6) -> str:
        if not candidates:
            return "No prior optimization history."
        ordered = sorted(candidates, key=lambda c: c.mean, reverse=True)[:limit]
        lines = ["Historical optimization context:"]
        for c in ordered:
            feedback = c.latest_feedback.replace("\n", " ")[:240]
            lines.append(f"- score={c.mean:.3f}; candidate={c.program}; feedback={feedback}")
        return "\n".join(lines)
