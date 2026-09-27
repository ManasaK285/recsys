import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# PATHS
# =========================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    ROOT,
    "data",
    "raw",
    "trustlens.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 70)
print("TrustLens Dataset Sanity Check")
print("=" * 70)

print(f"\nDataset: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"\nShape: {df.shape}")

print("\nColumns:")
for col in df.columns:
    print(f"  - {col}")


# =========================================================
# BASIC STATISTICS
# =========================================================

print("\n" + "=" * 70)
print("1. BASIC DATASET STATISTICS")
print("=" * 70)

print("\nMissing values:")
print(df.isnull().sum())

print(
    "\nDuplicate rows:",
    df.duplicated().sum()
)

print(
    "\nUnique response texts:",
    df["response"].nunique()
)

print(
    "Total response rows:",
    len(df)
)

print(
    "Duplicate response rows:",
    df["response"].duplicated().sum()
)


# =========================================================
# DUPLICATE RESPONSE ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("2. RESPONSE DUPLICATION")
print("=" * 70)

response_counts = (
    df["response"]
    .value_counts()
)

print(
    f"\nUnique responses: "
    f"{len(response_counts)}"
)

print(
    f"Responses occurring more than once: "
    f"{(response_counts > 1).sum()}"
)

print(
    f"Maximum repetitions of one response: "
    f"{response_counts.max()}"
)

print("\nTop repeated responses:")

for response, count in response_counts.head(10).items():

    print("\n--------------------------------")

    print(
        f"COUNT: {count}"
    )

    print(
        response
    )


# =========================================================
# LABEL DISTRIBUTION
# =========================================================

print("\n" + "=" * 70)
print("3. LABEL DISTRIBUTIONS")
print("=" * 70)

print("\nSource:")
print(
    df["source"]
    .value_counts()
)

print("\nAgreement:")
print(
    df["agreement"]
    .value_counts()
)


# =========================================================
# SOURCE + UNIQUE RESPONSE
# =========================================================

print("\n" + "=" * 70)
print("4. UNIQUE RESPONSES BY SOURCE")
print("=" * 70)

unique_by_source = (
    df.groupby("source")["response"]
    .nunique()
)

print(unique_by_source)


# =========================================================
# RESPONSE LENGTH
# =========================================================

print("\n" + "=" * 70)
print("5. RESPONSE LENGTH BY SOURCE")
print("=" * 70)

df["_response_length"] = (
    df["response"]
    .astype(str)
    .str.len()
)

df["_word_count"] = (
    df["response"]
    .astype(str)
    .str.split()
    .str.len()
)

print(
    df.groupby("source")[
        [
            "_response_length",
            "_word_count"
        ]
    ].agg(
        [
            "mean",
            "std",
            "min",
            "max"
        ]
    )
)


# =========================================================
# SAME RESPONSE / DIFFERENT SOURCE
# =========================================================

print("\n" + "=" * 70)
print("6. SAME RESPONSE WITH DIFFERENT SOURCE LABEL")
print("=" * 70)

response_source_counts = (
    df.groupby("response")["source"]
    .nunique()
)

conflicting_responses = (
    response_source_counts[
        response_source_counts > 1
    ]
)

print(
    "Responses appearing under multiple sources:",
    len(conflicting_responses)
)

if len(conflicting_responses) > 0:

    print("\nWARNING: conflicting labels detected.")

else:

    print(
        "No response has conflicting source labels."
    )


# =========================================================
# TF-IDF BASELINE
# =========================================================

print("\n" + "=" * 70)
print("7. TF-IDF BASELINE")
print("=" * 70)

X_text = (
    df["response"]
    .astype(str)
)

y = (
    df["source"]
    .astype(str)
)

X_train, X_test, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=10000
)

X_train_vec = vectorizer.fit_transform(
    X_train
)

X_test_vec = vectorizer.transform(
    X_test
)

clf = LogisticRegression(
    max_iter=2000
)

clf.fit(
    X_train_vec,
    y_train
)

predictions = clf.predict(
    X_test_vec
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"\nTF-IDF accuracy: {accuracy:.4f}"
)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


# =========================================================
# LABEL SHUFFLE
# =========================================================

print("\n" + "=" * 70)
print("8. LABEL-SHUFFLE SANITY TEST")
print("=" * 70)

shuffled_y = (
    y.sample(
        frac=1,
        random_state=123
    )
    .reset_index(drop=True)
)

X_reset = (
    X_text
    .reset_index(drop=True)
)

X_train, X_test, y_train, y_test = train_test_split(
    X_reset,
    shuffled_y,
    test_size=0.25,
    random_state=42,
    stratify=shuffled_y
)

vectorizer_shuffle = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=10000
)

X_train_vec = vectorizer_shuffle.fit_transform(
    X_train
)

X_test_vec = vectorizer_shuffle.transform(
    X_test
)

clf_shuffle = LogisticRegression(
    max_iter=2000
)

clf_shuffle.fit(
    X_train_vec,
    y_train
)

predictions_shuffle = clf_shuffle.predict(
    X_test_vec
)

shuffle_accuracy = accuracy_score(
    y_test,
    predictions_shuffle
)

print(
    f"\nAccuracy after label shuffling: "
    f"{shuffle_accuracy:.4f}"
)

print(
    "\nExpected: approximately 0.50 "
    "for binary classification."
)


# =========================================================
# TOP FEATURES
# =========================================================

print("\n" + "=" * 70)
print("9. MOST PREDICTIVE FEATURES")
print("=" * 70)

feature_names = np.array(
    vectorizer.get_feature_names_out()
)

print(
    "\nClasses:",
    clf.classes_
)

# Binary logistic regression has ONE coefficient vector.
# Positive = class clf.classes_[1]
# Negative = class clf.classes_[0]

if len(clf.classes_) == 2:

    coefficients = clf.coef_[0]

    positive_indices = np.argsort(
        coefficients
    )[-25:][::-1]

    negative_indices = np.argsort(
        coefficients
    )[:25]

    print(
        f"\nFeatures associated with "
        f"class: {clf.classes_[1]}"
    )

    for idx in positive_indices:

        print(
            f"  {feature_names[idx]:35s} "
            f"{coefficients[idx]: .4f}"
        )

    print(
        f"\nFeatures associated with "
        f"class: {clf.classes_[0]}"
    )

    for idx in negative_indices:

        print(
            f"  {feature_names[idx]:35s} "
            f"{coefficients[idx]: .4f}"
        )


# =========================================================
# SOURCE / SCENARIO TABLE
# =========================================================

print("\n" + "=" * 70)
print("10. SOURCE × SCENARIO")
print("=" * 70)

print(
    pd.crosstab(
        df["scenario_id"],
        df["source"]
    ).head(20)
)


# =========================================================
# SOURCE / SCENARIO TYPE
# =========================================================

print("\n" + "=" * 70)
print("11. SOURCE × SCENARIO TYPE")
print("=" * 70)

print(
    pd.crosstab(
        df["scenario_type"],
        df["source"]
    )
)


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 70)
print("SANITY CHECK COMPLETE")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "Your current dataset contains many repeated response texts."
)

print(
    "Before interpreting 1.00 source-detection performance,"
)

print(
    "we should inspect make_dataset.py and determine whether"
)

print(
    "the repeated responses are intentional experimental"
)

print(
    "conditions or an artifact of dataset construction."
)