"""
Experiment 05: Generalization

Tests whether source detection generalizes to unseen scenarios.

Evaluation:
1. Random train/test split (baseline)
2. Leave-One-Scenario-Out (LOSO)
   - Hold out one entire scenario
   - Train on all remaining scenarios
   - Test only on the unseen scenario

This helps determine whether the source detector is learning
general source/style cues or memorizing scenario-specific vocabulary.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "trustlens.csv"


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def evaluate_model(y_true, y_pred, y_prob):
    """Calculate classification metrics."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

print("=" * 80)
print("TRUSTLENS - EXPERIMENT 05: GENERALIZATION")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

print(f"\nLoaded dataset: {df.shape}")
print(f"Scenarios: {df['scenario'].nunique()}")
print(f"Sources: {df['source'].value_counts().to_dict()}")

# Convert labels to binary
# human = 0
# ai = 1
df["source_label"] = (df["source"] == "ai").astype(int)

X = df["response"].astype(str)
y = df["source_label"]


# ---------------------------------------------------------------------
# PART 1: Random split baseline
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("PART 1: RANDOM TRAIN/TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=10000,
    sublinear_tf=True,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

model.fit(X_train_tfidf, y_train)

random_pred = model.predict(X_test_tfidf)
random_prob = model.predict_proba(X_test_tfidf)[:, 1]

random_metrics = evaluate_model(
    y_test,
    random_pred,
    random_prob,
)

print("\nRandom split results:")

for metric, value in random_metrics.items():
    print(f"{metric:>10}: {value:.4f}")


# ---------------------------------------------------------------------
# PART 2: Leave-One-Scenario-Out
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("PART 2: LEAVE-ONE-SCENARIO-OUT GENERALIZATION")
print("=" * 80)

scenarios = sorted(df["scenario"].unique())

print(f"\nTesting {len(scenarios)} scenarios:")
print(", ".join(scenarios))

results = []

for held_out_scenario in scenarios:

    print("\n" + "-" * 80)
    print(f"Held-out scenario: {held_out_scenario}")
    print("-" * 80)

    # -------------------------------------------------------------
    # Split by scenario
    # -------------------------------------------------------------

    train_df = df[df["scenario"] != held_out_scenario].copy()
    test_df = df[df["scenario"] == held_out_scenario].copy()

    X_train = train_df["response"].astype(str)
    y_train = train_df["source_label"]

    X_test = test_df["response"].astype(str)
    y_test = test_df["source_label"]

    print(f"Train size: {len(train_df)}")
    print(f"Test size:  {len(test_df)}")

    # -------------------------------------------------------------
    # TF-IDF
    # -------------------------------------------------------------

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # -------------------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    model.fit(X_train_tfidf, y_train)

    pred = model.predict(X_test_tfidf)
    prob = model.predict_proba(X_test_tfidf)[:, 1]

    metrics = evaluate_model(
        y_test,
        pred,
        prob,
    )

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    results.append(
        {
            "held_out_scenario": held_out_scenario,
            "n_test": len(test_df),
            **metrics,
        }
    )


# ---------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("LEAVE-ONE-SCENARIO-OUT RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ---------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------

metric_columns = [
    "accuracy",
    "macro_f1",
    "precision",
    "recall",
    "roc_auc",
]

mean_metrics = results_df[metric_columns].mean()
std_metrics = results_df[metric_columns].std()

print("\n" + "=" * 80)
print("GENERALIZATION SUMMARY")
print("=" * 80)

for metric in metric_columns:
    print(
        f"{metric:>10}: "
        f"{mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}"
    )


# ---------------------------------------------------------------------
# Compare random split vs LOSO
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("RANDOM SPLIT VS UNSEEN-SCENARIO GENERALIZATION")
print("=" * 80)

comparison = pd.DataFrame(
    {
        "evaluation": [
            "Random split",
            "Leave-one-scenario-out",
        ],
        "accuracy": [
            random_metrics["accuracy"],
            mean_metrics["accuracy"],
        ],
        "macro_f1": [
            random_metrics["macro_f1"],
            mean_metrics["macro_f1"],
        ],
        "precision": [
            random_metrics["precision"],
            mean_metrics["precision"],
        ],
        "recall": [
            random_metrics["recall"],
            mean_metrics["recall"],
        ],
        "roc_auc": [
            random_metrics["roc_auc"],
            mean_metrics["roc_auc"],
        ],
    }
)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

output_path = OUTPUT_DIR / "05_generalization.csv"

results_df.to_csv(
    output_path,
    index=False,
)

print(f"\nSaved scenario-level results to:")
print(output_path)

print("\nExperiment 05 completed successfully.")