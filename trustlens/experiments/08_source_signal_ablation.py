"""
TRUSTLENS - EXPERIMENT 08
SOURCE-SIGNAL ABLATION

Final diagnostic experiment.

Goal:
Identify which parts of the synthetic response contain source-specific
information by progressively removing source-specific components.

Conditions:
1. Full response
2. Style neutralized
3. Frame neutralized
4. Decision phrase neutralized
5. Style + frame neutralized
6. Style + frame + decision neutralized

Each condition uses the same TF-IDF + Logistic Regression classifier.
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


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "trustlens.csv"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Shared neutral replacements
# ---------------------------------------------------------------------

NEUTRAL_STYLES = [
    "This point is relevant to the decision.",
    "That consideration should be included in the analysis.",
    "This factor is worth taking into account.",
    "This is one part of the situation that matters.",
    "That aspect should also be considered.",
]

NEUTRAL_FRAME = (
    "This perspective is relevant to the situation."
)

NEUTRAL_DECISION = (
    "Overall, this consideration should be included in the decision."
)


# ---------------------------------------------------------------------
# Original source-specific phrases
# ---------------------------------------------------------------------

HUMAN_STYLES = [
    "Personally, I would put some weight on that.",
    "From my perspective, that matters.",
    "I would be hesitant to ignore that.",
    "I think that is worth keeping in mind.",
    "That seems important to me.",
]

AI_STYLES = [
    "This also suggests that the competing consideration should not be ignored.",
    "The competing consideration should therefore remain part of the analysis.",
    "This makes the tradeoff between the relevant factors important.",
    "The alternative should also be considered before reaching a conclusion.",
    "This indicates that both sides of the decision should be evaluated.",
]


# ---------------------------------------------------------------------
# Extract source-specific phrase candidates from dataset
# ---------------------------------------------------------------------

def get_source_phrases(df, column):
    """
    Return unique phrases used in a source-specific metadata column.

    The generator stores the actual template ID, while the response
    contains the corresponding natural-language phrase.

    We therefore maintain explicit phrase lists above for style and
    discover frame/decision phrases from the response text where possible.
    """

    return sorted(
        df[column].dropna().astype(str).unique()
    )


# ---------------------------------------------------------------------
# Replace phrases
# ---------------------------------------------------------------------

def replace_phrases(text, phrases, replacement):
    """
    Replace all supplied phrases in a response.
    """
    for phrase in phrases:
        text = text.replace(
            str(phrase),
            replacement,
        )

    return text


def neutralize_style(text):
    """
    Remove the explicit human/AI style phrase.
    """

    text = replace_phrases(
        text,
        HUMAN_STYLES,
        NEUTRAL_STYLES[0],
    )

    text = replace_phrases(
        text,
        AI_STYLES,
        NEUTRAL_STYLES[0],
    )

    return text


# ---------------------------------------------------------------------
# Detect source-specific frame / decision text
# ---------------------------------------------------------------------

def discover_source_phrases(df, column):
    """
    Discover phrases associated with source-specific metadata.

    For each template ID, find the most common repeated sentence-like
    fragment across responses.

    We use token-level candidate extraction rather than hard-coding
    every phrase.
    """

    phrases = []

    for template_id in sorted(
        df[column].dropna().unique()
    ):

        subset = df[
            df[column] == template_id
        ]

        if subset.empty:
            continue

        # Use the most common sentence fragment shared by responses.
        # Responses are generated from templates, so repeated fragments
        # should occur frequently.
        texts = subset["response"].astype(str)

        # Count complete sentences.
        sentence_counts = {}

        for text in texts:

            sentences = [
                sentence.strip()
                for sentence in text.split(".")
                if sentence.strip()
            ]

            for sentence in sentences:

                if len(sentence.split()) >= 4:

                    sentence = sentence + "."

                    sentence_counts[sentence] = (
                        sentence_counts.get(sentence, 0) + 1
                    )

        if sentence_counts:

            best_sentence = max(
                sentence_counts,
                key=sentence_counts.get,
            )

            # Only treat it as a template phrase if repeated.
            if sentence_counts[best_sentence] >= 2:
                phrases.append(best_sentence)

    return phrases


# ---------------------------------------------------------------------
# Build ablated responses
# ---------------------------------------------------------------------

def build_condition(df, condition):
    """
    Construct response text for a particular ablation condition.
    """

    result = df.copy()

    result["text"] = result["response"].astype(str)

    # ---------------------------------------------------------------
    # Style neutralization
    # ---------------------------------------------------------------

    if "style" in condition:

        result["text"] = result["text"].apply(
            neutralize_style
        )

    # ---------------------------------------------------------------
    # Frame neutralization
    # ---------------------------------------------------------------

    if "frame" in condition:

        frame_phrases = discover_source_phrases(
            result,
            "frame_id",
        )

        result["text"] = result["text"].apply(
            lambda x: replace_phrases(
                x,
                frame_phrases,
                NEUTRAL_FRAME,
            )
        )

    # ---------------------------------------------------------------
    # Decision neutralization
    # ---------------------------------------------------------------

    if "decision" in condition:

        decision_phrases = discover_source_phrases(
            result,
            "decision_id",
        )

        result["text"] = result["text"].apply(
            lambda x: replace_phrases(
                x,
                decision_phrases,
                NEUTRAL_DECISION,
            )
        )

    return result


# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------

def evaluate(df):

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["source_label"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["source_label"],
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

    model.fit(
        X_train_vec,
        y_train,
    )

    predictions = model.predict(
        X_test_vec
    )

    probabilities = model.predict_proba(
        X_test_vec
    )[:, 1]

    return {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
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
    print("TRUSTLENS - EXPERIMENT 08")
    print("SOURCE-SIGNAL ABLATION")
    print("=" * 80)

    df = pd.read_csv(
        DATA_PATH
    )

    df["source_label"] = (
        df["source"] == "ai"
    ).astype(int)

    print(
        f"\nDataset shape: {df.shape}"
    )

    print("\nSource distribution:")
    print(
        df["source"].value_counts()
    )

    # ---------------------------------------------------------------
    # Conditions
    # ---------------------------------------------------------------

    conditions = [
        (
            "full_response",
            [],
        ),
        (
            "style_neutralized",
            ["style"],
        ),
        (
            "frame_neutralized",
            ["frame"],
        ),
        (
            "decision_neutralized",
            ["decision"],
        ),
        (
            "style_frame_neutralized",
            ["style", "frame"],
        ),
        (
            "style_frame_decision_neutralized",
            ["style", "frame", "decision"],
        ),
    ]

    results = []

    # ---------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------

    for name, components in conditions:

        print("\n" + "=" * 80)
        print(
            f"CONDITION: {name}"
        )
        print("=" * 80)

        condition_df = build_condition(
            df,
            components,
        )

        metrics = evaluate(
            condition_df
        )

        row = {
            "condition": name,
            **metrics,
        }

        results.append(row)

        for metric, value in metrics.items():
            print(
                f"{metric:>10}: {value:.4f}"
            )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    output_path = (
        RESULTS_DIR
        / "08_source_signal_ablation.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXPERIMENT 08 SUMMARY")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nResults saved to:")
    print(output_path)

    print("\n" + "=" * 80)
    print("EXPERIMENT 08 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()