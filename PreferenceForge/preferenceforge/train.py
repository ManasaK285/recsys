from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from preferenceforge.data import load_dataset
from preferenceforge.models import GlobalRewardModel, IndependentRewardModel, AdaptiveRewardFeatureModel, SupportSummaryModel

def split_data(df, seed=42):
    # Split within each user to ensure each user contributes train and held-out examples.
    trains, tests = [], []
    for _, group in df.groupby("user_id"):
        tr, te = train_test_split(group, test_size=0.25, random_state=seed, stratify=group["label"] if group["label"].nunique() > 1 else None)
        trains.append(tr); tests.append(te)
    return pd.concat(trains).sort_index(), pd.concat(tests).sort_index()

def train_all(data_path="data/preferences.csv"):
    df = load_dataset(data_path)
    train, test = split_data(df)
    models = {
        "global": GlobalRewardModel().fit(train),
        "independent": IndependentRewardModel().fit(train),
        "adaptive_reward_features": AdaptiveRewardFeatureModel().fit(train, support_size=8),
        "support_summary": SupportSummaryModel().fit(train, support_size=8),
    }
    Path("artifacts").mkdir(exist_ok=True)
    payload = {
        "train_rows": len(train), "test_rows": len(test),
        "models": {name: model.export() for name, model in models.items()},
        "test_indices": test.index.tolist(),
    }
    Path("artifacts/models.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Persist split indices so evaluation uses exactly the same holdout.
    train.to_csv("artifacts/train.csv", index=False)
    test.to_csv("artifacts/test.csv", index=False)
    return models, train, test
