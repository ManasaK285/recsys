"""Trains the reason-quality classifier and regressor, saves them with joblib."""
import os
import joblib
from sklearn.model_selection import train_test_split

from dataset import build_dataset
from reason_quality import (
    build_classifier_pipeline,
    build_quality_regressor_pipeline,
)

HERE = os.path.dirname(__file__)


def train_all(out_dir: str = None, seed: int = 42):
    out_dir = out_dir or os.path.join(HERE, "..", "data", "results")
    os.makedirs(out_dir, exist_ok=True)

    df = build_dataset(seed=seed)
    X = df["reason_text"]
    y_class = df["reason_type"]
    y_reg = df["quality_score"]

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_class, y_reg, test_size=0.25, random_state=seed, stratify=y_class
    )

    clf = build_classifier_pipeline()
    clf.fit(X_train, yc_train)

    reg = build_quality_regressor_pipeline()
    reg.fit(X_train, yr_train)

    joblib.dump(clf, os.path.join(out_dir, "reason_type_classifier.joblib"))
    joblib.dump(reg, os.path.join(out_dir, "quality_regressor.joblib"))

    # Persist the split so evaluate.py scores on the same held-out set
    split = {
        "X_test": X_test.tolist(),
        "yc_test": yc_test.tolist(),
        "yr_test": yr_test.tolist(),
    }
    joblib.dump(split, os.path.join(out_dir, "test_split.joblib"))

    print(f"[train] saved models and test split to {out_dir}")
    return clf, reg, split


if __name__ == "__main__":
    train_all()
