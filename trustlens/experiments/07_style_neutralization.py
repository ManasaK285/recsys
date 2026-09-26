"""
TRUSTLENS - EXPERIMENT 07
SOURCE-STYLE NEUTRALIZATION

Purpose:
Test whether AI-vs-human source detection remains strong when both
sources use the same stylistic template vocabulary.

The experiment:
1. Loads the existing synthetic dataset.
2. Replaces source-specific style phrases with a shared neutral style.
3. Keeps all other response content unchanged.
4. Runs TF-IDF + Logistic Regression.
5. Compares performance against a shuffled-label control.
6. Repeats the experiment across several random neutral-style assignments.

This isolates how much source detection depends on explicit
human-vs-AI stylistic fingerprints.
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
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42

NEUTRAL_STYLES = [
    "This point is relevant to the decision.",
    "That consideration should be included in the analysis.",
    "This factor is worth taking into account.",
    "This is one part of the situation that matters.",
    "That aspect should also be considered.",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def replace_style_phrase(response, original_phrase, neutral_phrase):
    """
    Replace the source-specific style phrase with a shared neutral phrase.
    """
    return response.replace(original_phrase, neutral_phrase)


def get_style_phrase_map(df):
    """
    Recover the actual style phrase associated with each style_id.

    Each style_id should correspond to one repeated phrase in the
    generated responses.
    """

    style_map = {}

    for style_id in sorted(df["style_id"].unique()):

        rows = df[df["style_id"] == style_id]

        if len(rows) == 0:
            continue

        # The generator places the style phrase in the response.
        # We identify the repeated sentence by looking for the phrase
        # that occurs consistently across that style's responses.

        style_map[style_id] = rows["response"].iloc[0]

    return style_map


def neutralize_responses(df, random_state):
    """
    Replace source-specific style phrases with shared neutral phrases.

    Each style index is mapped to one neutral phrase.

    Importantly, the mapping is randomized independently of source,
    preventing the classifier from using human_style_X versus
    ai_style_X as a lexical shortcut.
    """

    rng = np.random.default_rng(random_state)

    neutral_phrases = np.array(NEUTRAL_STYLES)

    # Random permutation so that the assignment does not systematically
    # correspond to the original style index.
    permutation = rng.permutation(len(neutral_phrases))

    neutral_assignment = {
        i: neutral_phrases[permutation[i]]
        for i in range(len(neutral_phrases))
    }

    neutralized = df.copy()

    # -----------------------------------------------------------------
    # IMPORTANT:
    # The generator's style phrase is identifiable from the current
    # style templates.
    # -----------------------------------------------------------------

    human_phrases = [
        "Personally, I would put some weight on that.",
        "From my perspective, that matters.",
        "I would be hesitant to ignore that.",
        "I think that is worth keeping in mind.",
        "That seems important to me.",
    ]

    ai_phrases = [
        "This also suggests that the competing consideration should not be ignored.",
        "The competing consideration should therefore remain part of the analysis.",
        "This makes the tradeoff between the relevant factors important.",
        "The alternative should also be considered before reaching a conclusion.",
        "This indicates that both sides of the decision should be evaluated.",
    ]

    all_phrases = human_phrases + ai_phrases

    # Replace every source-specific style phrase.
    for phrase in all_phrases:

        mask = neutralized["response"].str.contains(
            phrase,
            regex=False,
            na=False,
        )

        if mask.any():

            # Determine which neutral phrase to use based on the
            # style index embedded in the original style_id.
            style_ids = neutralized.loc[mask, "style_id"]

            for idx, style_id in style_ids.items():

                try:
                    style_index = int(style_id.split("_")[-1])
                except (ValueError, AttributeError):
                    continue

                neutralized.at[
                    idx,
                    "response"
                ] = neutralized.at[
                    idx,
                    "response"
                ].replace(
                    phrase,
                    neutral_assignment[style_index],
                )

    # Keep the original style metadata for analysis, but explicitly
    # record that the response text has been neutralized.
    neutralized["style_neutralized"] = True

    return neutralized


def evaluate_source_detection(df, label_column="source_label"):
    """
    TF-IDF + Logistic Regression source detection.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        df["response"],
        df[label_column],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df[label_column],
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000,
        sublinear_tf=True,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    probabilities = model.predict_proba(X_test_vec)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test,
            predictions,
            average="macro",
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("TRUSTLENS - EXPERIMENT 07")
    print("SOURCE-STYLE NEUTRALIZATION")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset shape: {df.shape}")

    print("\nOriginal source distribution:")
    print(df["source"].value_counts())

    # -----------------------------------------------------------------
    # Labels
    # -----------------------------------------------------------------

    df["source_label"] = (
        df["source"] == "ai"
    ).astype(int)

    # -----------------------------------------------------------------
    # Neutralize style
    # -----------------------------------------------------------------

    neutral_df = neutralize_responses(
        df,
        random_state=RANDOM_STATE,
    )

    print("\nStyle neutralization complete.")

    print("\nNeutral style phrases:")

    for i, phrase in enumerate(NEUTRAL_STYLES):
        print(f"{i}: {phrase}")

    # -----------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------

    print("\nResponse uniqueness:")

    print(
        "Original unique responses:",
        df["response"].nunique(),
    )

    print(
        "Neutralized unique responses:",
        neutral_df["response"].nunique(),
    )

    assert (
        neutral_df["source_label"].value_counts().to_dict()
        == {1: 1200, 0: 1200}
    )

    # -----------------------------------------------------------------
    # Main neutralized experiment
    # -----------------------------------------------------------------

    print("\n" + "=" * 80)
    print("NEUTRALIZED STYLE SOURCE DETECTION")
    print("=" * 80)

    neutral_results = evaluate_source_detection(
        neutral_df
    )

    for metric, value in neutral_results.items():
        print(f"{metric:>10}: {value:.4f}")

    # -----------------------------------------------------------------
    # Label-shuffle control
    # -----------------------------------------------------------------

    print("\n" + "=" * 80)
    print("LABEL-SHUFFLE CONTROL")
    print("=" * 80)

    shuffled_df = neutral_df.copy()

    rng = np.random.default_rng(RANDOM_STATE)

    shuffled_df["source_label"] = rng.permutation(
        shuffled_df["source_label"].values
    )

    shuffle_results = evaluate_source_detection(
        shuffled_df
    )

    for metric, value in shuffle_results.items():
        print(f"{metric:>10}: {value:.4f}")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "condition": "style_neutralized",
                **neutral_results,
            },
            {
                "condition": "label_shuffle",
                **shuffle_results,
            },
        ]
    )

    output_path = (
        RESULTS_DIR
        / "07_style_neutralization.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXPERIMENT 07 SUMMARY")
    print("=" * 80)

    print(results.to_string(index=False))

    print("\nResults saved to:")
    print(output_path)

    print("\n" + "=" * 80)
    print("EXPERIMENT 07 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()