from preferenceforge.data import generate_dataset
from preferenceforge.features import pairwise_features

def test_pairwise_feature_shape():
    df, _ = generate_dataset(users=2, pairs_per_user=5)
    X = pairwise_features(df)
    assert X.shape == (10, 5)
