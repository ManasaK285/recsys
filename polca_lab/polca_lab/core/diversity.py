import math
import re
from collections import Counter
from typing import List


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def tfidf_vector(texts: List[str]) -> List[List[float]]:
    tokenized = [tokenize(t) for t in texts]
    vocab = sorted(set(tok for row in tokenized for tok in row))
    if not vocab:
        return [[0.0] for _ in texts]
    df = Counter()
    for row in tokenized:
        df.update(set(row))
    n = len(texts)
    vectors = []
    for row in tokenized:
        counts = Counter(row)
        norm = max(len(row), 1)
        vectors.append([
            (counts[w] / norm) * (math.log((n + 1) / (df[w] + 1)) + 1.0)
            for w in vocab
        ])
    return vectors


def cosine_distance(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


class EpsilonNet:
    """Reward-free semantic admission filter.

    A candidate is admitted only when its distance from every retained
    candidate exceeds epsilon. This keeps memory diverse without using reward.
    """
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def accept(self, candidate_text: str, memory_texts: List[str]) -> bool:
        if not memory_texts:
            return True
        vectors = tfidf_vector(memory_texts + [candidate_text])
        new_vec = vectors[-1]
        return all(cosine_distance(new_vec, old) > self.epsilon for old in vectors[:-1])
