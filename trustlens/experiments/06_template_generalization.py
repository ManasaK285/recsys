"""
Experiment 06: Proper Paired Style-Template Generalization

Tests whether source detection generalizes to completely unseen
source-specific style templates.

For each fold:

    TRAIN:
        human_style_0,1,2,3
        ai_style_0,1,2,3

    TEST:
        human_style_4
        ai_style_4

Then repeat for every style index.

This prevents the previous experiment's problem where human and
AI templates shared the same numeric style ID.
"""

from pathlib import Path

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


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "trustlens.csv"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("TRUSTLENS - EXPERIMENT 06")
print("PROPER PAIRED STYLE-TEMPLATE GENERALIZATION")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

df["source_label"] = (
    df["source"] == "ai"
).astype(int)

print(f"\nDataset shape: {df.shape}")

print("\nSource distribution:")
print(df["source"].value_counts())

print("\nStyle templates by source:")

print(
    df.groupby("source")["style_id"]
    .nunique()
)


# ============================================================
# CHECK TEMPLATE STRUCTURE
# ============================================================

human_styles = sorted(
    df.loc[
        df["source"] == "human",
        "style_id",
    ].unique()
)

ai_styles = sorted(
    df.loc[
        df["source"] == "ai",
        "style_id",
    ].unique()
)

print("\nHuman style templates:")
print(human_styles)

print("\nAI style templates:")
print(ai_styles)

assert len(human_styles) == 5
assert len(ai_styles) == 5


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
    probabilities,
):
    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
    }


# ============================================================
# PAIRED STYLE HOLDOUT
# ============================================================

results = []

# We expect:
#
# human_style_0
# human_style_1
# ...
#
# ai_style_0
# ai_style_1
# ...

for style_index in range(5):

    human_style = f"human_style_{style_index}"
    ai_style = f"ai_style_{style_index}"

    print("\n" + "=" * 80)

    print(
        f"HOLDING OUT STYLE PAIR {style_index}"
    )

    print(
        f"Human template: {human_style}"
    )

    print(
        f"AI template   : {ai_style}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Test = BOTH unseen source-specific templates
    # --------------------------------------------------------

    test_mask = (
        (df["style_id"] == human_style)
        |
        (df["style_id"] == ai_style)
    )

    train_df = df[
        ~test_mask
    ].copy()

    test_df = df[
        test_mask
    ].copy()

    print(
        f"\nTrain size: {len(train_df)}"
    )

    print(
        f"Test size : {len(test_df)}"
    )

    print("\nTest source distribution:")

    print(
        test_df["source"].value_counts()
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(
        train_df["response"].astype(str)
    )

    X_test = vectorizer.transform(
        test_df["response"].astype(str)
    )

    y_train = train_df["source_label"]

    y_test = test_df["source_label"]

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\nAccuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1 : {metrics['macro_f1']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"ROC-AUC  : {metrics['roc_auc']:.4f}"
    )

    results.append(
        {
            "held_out_style_index": style_index,
            "held_out_human_style": human_style,
            "held_out_ai_style": ai_style,
            "n_train": len(train_df),
            "n_test": len(test_df),
            **metrics,
        }
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    RESULTS_DIR
    / "06_paired_style_generalization.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PAIRED STYLE-TEMPLATE GENERALIZATION SUMMARY")
print("=" * 80)

metrics = [
    "accuracy",
    "macro_f1",
    "precision",
    "recall",
    "roc_auc",
]

for metric in metrics:

    mean = results_df[metric].mean()

    std = results_df[metric].std()

    print(
        f"{metric:>10}: "
        f"{mean:.4f} ± {std:.4f}"
    )


# ============================================================
# DETAILED TABLE
# ============================================================

print("\n" + "=" * 80)
print("FOLD RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 06 COMPLETE")
print("=" * 80)

print(
    f"\nResults saved to:\n{output_path}"
)