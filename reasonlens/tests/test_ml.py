import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "ml"))

from dataset import build_dataset  # noqa: E402
from reason_quality import build_classifier_pipeline, build_quality_regressor_pipeline  # noqa: E402
from evaluate import evaluate_classifier, evaluate_regressor  # noqa: E402


def test_dataset_has_expected_columns():
    df = build_dataset(n_per_type=10, seed=1)
    for col in [
        "reason_type",
        "reason_text",
        "specificity",
        "necessity",
        "transparency",
        "advertising_disclosure",
        "privacy_language",
        "quality_score",
    ]:
        assert col in df.columns
    assert len(df) == 50  # 5 reason types x 10


def test_quality_score_in_range():
    df = build_dataset(n_per_type=20, seed=2)
    assert df["quality_score"].between(0, 1).all()


def test_classifier_pipeline_trains_and_predicts():
    df = build_dataset(n_per_type=30, seed=3)
    clf = build_classifier_pipeline()
    clf.fit(df["reason_text"], df["reason_type"])
    preds = clf.predict(df["reason_text"].iloc[:5])
    assert len(preds) == 5

    metrics = evaluate_classifier(clf, df["reason_text"], df["reason_type"], sorted(df["reason_type"].unique()))
    assert 0 <= metrics["accuracy"] <= 1


def test_regressor_pipeline_trains_and_predicts():
    df = build_dataset(n_per_type=30, seed=4)
    reg = build_quality_regressor_pipeline()
    reg.fit(df["reason_text"], df["quality_score"])
    preds = reg.predict(df["reason_text"].iloc[:5])
    assert len(preds) == 5

    metrics = evaluate_regressor(reg, df["reason_text"], df["quality_score"])
    assert "mae" in metrics and "r2" in metrics
