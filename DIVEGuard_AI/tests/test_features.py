import pandas as pd
from src.diveguard.features import prepare_features

def test_global_features_do_not_include_demographics():
    df = pd.DataFrame({
        "text_prompt": ["hello"], "topic": ["Bias/Gender"],
        "rater_gender": ["woman"], "rater_ethnicity": ["southasian"]
    })
    x = prepare_features(df, pluralistic=False)
    assert list(x.columns) == ["text"]

def test_pluralistic_features_include_context():
    df = pd.DataFrame({
        "text_prompt": ["hello"], "topic": ["Bias/Gender"],
        "rater_gender": ["woman"], "rater_ethnicity": ["southasian"]
    })
    x = prepare_features(df, pluralistic=True)
    assert "rater_gender" in x.columns
    assert "context_topic" in x.columns
