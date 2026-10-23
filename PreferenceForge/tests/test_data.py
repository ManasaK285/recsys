from preferenceforge.data import generate_dataset
from preferenceforge.config import FEATURES

def test_generate_dataset_shape():
    df, weights = generate_dataset(users=3, pairs_per_user=10, seed=1)
    assert len(df) == 30
    assert set(df["label"].unique()).issubset({0, 1})
    assert all(f"a_{x}" in df.columns for x in FEATURES)
    assert len(weights) == 3
