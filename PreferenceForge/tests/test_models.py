from preferenceforge.data import generate_dataset
from preferenceforge.train import split_data
from preferenceforge.models import GlobalRewardModel, AdaptiveRewardFeatureModel

def test_global_model_predicts_probabilities():
    df, _ = generate_dataset(users=4, pairs_per_user=30, seed=4)
    train, test = split_data(df)
    model = GlobalRewardModel().fit(train)
    p = model.predict_proba(test)
    assert len(p) == len(test)
    assert ((p >= 0) & (p <= 1)).all()

def test_adaptive_model_support():
    df, _ = generate_dataset(users=4, pairs_per_user=30, seed=5)
    train, test = split_data(df)
    model = AdaptiveRewardFeatureModel().fit(train, support_size=5)
    assert len(model.predict_proba(test)) == len(test)
