import os
import pandas as pd

from sklearn.model_selection import train_test_split


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

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("TrustLens Leakage Check")
print("=" * 70)

print(f"\nRows: {len(df)}")


# ---------------------------------------------------------
# Split
# ---------------------------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    random_state=42,
    stratify=df["source"]
)

print(f"\nTrain rows: {len(train_df)}")
print(f"Test rows:  {len(test_df)}")


# ---------------------------------------------------------
# Exact response overlap
# ---------------------------------------------------------

train_responses = set(
    train_df["response"]
    .astype(str)
)

test_responses = set(
    test_df["response"]
    .astype(str)
)

overlap = train_responses.intersection(
    test_responses
)

print("\n" + "=" * 70)
print("EXACT RESPONSE OVERLAP")
print("=" * 70)

print(
    f"\nUnique train responses: "
    f"{len(train_responses)}"
)

print(
    f"Unique test responses: "
    f"{len(test_responses)}"
)

print(
    f"Exact responses appearing in BOTH: "
    f"{len(overlap)}"
)

if len(overlap) > 0:

    print("\nWARNING: DATA LEAKAGE POSSIBLE")

    for response in list(overlap)[:10]:

        print("\n---")
        print(response[:500])

else:

    print(
        "\nNo exact response overlap detected."
    )


# ---------------------------------------------------------
# Scenario overlap
# ---------------------------------------------------------

if "scenario" in df.columns:

    train_scenarios = set(
        train_df["scenario"]
        .astype(str)
    )

    test_scenarios = set(
        test_df["scenario"]
        .astype(str)
    )

    scenario_overlap = (
        train_scenarios
        .intersection(test_scenarios)
    )

    print("\n" + "=" * 70)
    print("SCENARIO OVERLAP")
    print("=" * 70)

    print(
        f"\nTrain scenarios: "
        f"{len(train_scenarios)}"
    )

    print(
        f"Test scenarios: "
        f"{len(test_scenarios)}"
    )

    print(
        f"Scenarios in both: "
        f"{len(scenario_overlap)}"
    )

    if len(scenario_overlap) > 0:

        print(
            "\nNOTE:"
            "\nThe same scenarios appearing in both"
            "\ntrain and test are not necessarily leakage,"
            "\nbut they can make the task easier."
        )


# ---------------------------------------------------------
# Same scenario + source combinations
# ---------------------------------------------------------

if (
    "scenario" in df.columns
    and "source" in df.columns
):

    train_pairs = set(
        zip(
            train_df["scenario"].astype(str),
            train_df["source"].astype(str)
        )
    )

    test_pairs = set(
        zip(
            test_df["scenario"].astype(str),
            test_df["source"].astype(str)
        )
    )

    pair_overlap = train_pairs.intersection(
        test_pairs
    )

    print("\n" + "=" * 70)
    print("SCENARIO + SOURCE OVERLAP")
    print("=" * 70)

    print(
        f"\nPairs in train: {len(train_pairs)}"
    )

    print(
        f"Pairs in test: {len(test_pairs)}"
    )

    print(
        f"Pairs appearing in both: "
        f"{len(pair_overlap)}"
    )


print("\n" + "=" * 70)
print("LEAKAGE CHECK COMPLETE")
print("=" * 70)