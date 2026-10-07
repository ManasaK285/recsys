from __future__ import annotations

import re
from .models import Scenario


class RequirementCritic:
    """Checks whether a generated item obeys the requested mechanism."""

    def evaluate(self, s: Scenario) -> tuple[float, list[str]]:
        issues = []

        if not s.instruction.strip():
            issues.append("empty_instruction")

        if s.expected_label != s.intent:
            issues.append("label_intent_mismatch")

        for key, value in s.factors.items():
            if value not in s.instruction and key != "intent":
                # Some factors can be expressed semantically rather than
                # literally; therefore this is a soft signal.
                pass

        if len(s.instruction.split()) < 12:
            issues.append("too_short")

        score = max(0.0, 1.0 - 0.2 * len(issues))
        return score, issues


class IndependentCritic:
    """
    Second, structurally independent gate.

    This deliberately does not reuse the first critic's exact logic.
    """

    def evaluate(self, s: Scenario) -> tuple[float, list[str]]:
        issues = []

        if s.answer.strip() == "":
            issues.append("missing_answer")

        if s.complexity == "hard":
            if "constraint" not in s.instruction.lower():
                issues.append("hard_item_not_complex")

        if len(set(re.findall(r"\w+", s.instruction.lower()))) < 8:
            issues.append("low_lexical_variety")

        score = max(0.0, 1.0 - 0.25 * len(issues))
        return score, issues


def run_critics(s: Scenario, threshold: float = 0.75) -> Scenario:
    c1 = RequirementCritic()
    c2 = IndependentCritic()

    score1, issues1 = c1.evaluate(s)
    score2, issues2 = c2.evaluate(s)

    s.quality_score = round((score1 + score2) / 2, 3)
    s.critique_1 = ";".join(issues1) if issues1 else "PASS"
    s.critique_2 = ";".join(issues2) if issues2 else "PASS"
    s.accepted = score1 >= threshold and score2 >= threshold
    return s
