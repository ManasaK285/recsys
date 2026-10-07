from __future__ import annotations

from collections import Counter
from typing import Iterable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Scenario


def coverage(scenarios: Iterable[Scenario], factor_name: str) -> dict:
    values = Counter(
        s.factors.get(factor_name)
        for s in scenarios
        if s.accepted
    )
    return dict(values)


def duplicate_rate(scenarios: list[Scenario]) -> float:
    texts = [s.instruction.strip().lower() for s in scenarios if s.accepted]
    if not texts:
        return 0.0
    return 1.0 - len(set(texts)) / len(texts)


def diversity_score(scenarios: list[Scenario]) -> float:
    texts = [s.instruction for s in scenarios if s.accepted]
    if len(texts) < 2:
        return 0.0

    X = TfidfVectorizer(stop_words="english").fit_transform(texts)
    sim = cosine_similarity(X)
    upper = sim[np.triu_indices_from(sim, k=1)]
    if len(upper) == 0:
        return 0.0

    # 1 - average pairwise similarity
    return round(float(1.0 - upper.mean()), 3)


def dataset_report(scenarios: list[Scenario]) -> dict:
    accepted = [s for s in scenarios if s.accepted]
    return {
        "generated": len(scenarios),
        "accepted": len(accepted),
        "acceptance_rate": round(len(accepted) / max(len(scenarios), 1), 3),
        "duplicate_rate": round(duplicate_rate(scenarios), 3),
        "diversity_score": diversity_score(scenarios),
        "intent_coverage": coverage(scenarios, "intent"),
        "urgency_coverage": coverage(scenarios, "urgency"),
        "complexity_coverage": coverage(scenarios, "complexity"),
        "mean_quality": round(
            float(np.mean([s.quality_score for s in accepted])) if accepted else 0.0,
            3,
        ),
    }
