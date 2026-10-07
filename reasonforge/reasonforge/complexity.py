from __future__ import annotations

from .models import Scenario


LEVELS = {"easy": 1, "medium": 2, "hard": 3}


def score_complexity(s: Scenario) -> float:
    """
    Transparent proxy complexity score.

    In a production research setup this could be replaced by an independent
    model-based complexity judge or calibrated benchmark.
    """
    length_component = min(len(s.instruction.split()) / 80.0, 1.0)
    level_component = LEVELS.get(s.complexity, 1) / 3.0

    score = 0.45 * length_component + 0.55 * level_component
    return round(score, 3)
