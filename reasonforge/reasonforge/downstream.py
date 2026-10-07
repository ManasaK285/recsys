from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


@dataclass
class DownstreamResult:
    accuracy: float
    macro_f1: float


def train_and_evaluate(train_rows, test_rows) -> DownstreamResult:
    """
    Tests the actual utility of synthetic data.

    The synthetic set is the training set; the held-out gold set is never used
    during generation.
    """
    train_x = [x["instruction"] for x in train_rows]
    train_y = [x["label"] for x in train_rows]
    test_x = [x["instruction"] for x in test_rows]
    test_y = [x["label"] for x in test_rows]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    model.fit(train_x, train_y)
    pred = model.predict(test_x)

    return DownstreamResult(
        accuracy=round(float(accuracy_score(test_y, pred)), 3),
        macro_f1=round(float(f1_score(test_y, pred, average="macro")), 3),
    )
