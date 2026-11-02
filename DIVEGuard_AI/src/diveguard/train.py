from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, cohen_kappa_score
from .config import ARTIFACT_DIR, TARGET_DEFAULT, MIN_GROUP_SIZE
from .data import load_raw, clean_data, load_processed, save_processed
from .features import build_model, prepare_features, available_demographics

def split_by_image(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["image_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

def metric_pack(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "n": int(len(y_true)),
    }

def train(target: str = TARGET_DEFAULT, data_path: str | None = None):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if data_path:
        raw = load_raw(data_path)
        df = clean_data(raw, target)
    else:
        df = load_processed(target)
        if df is None:
            raw = load_raw(download=False)
            df = clean_data(raw, target)
            save_processed(df, target)
    if len(df) < 100:
        raise ValueError("Not enough labeled rows to train a meaningful model.")
    train_df, test_df = split_by_image(df)
    y_train, y_test = train_df[target], test_df[target]
    results = {}
    for name, pluralistic in [("global", False), ("pluralistic", True)]:
        x_train = prepare_features(train_df, pluralistic)
        x_test = prepare_features(test_df, pluralistic)
        cat_cols = [c for c in x_train.columns if c != "text"]
        model = build_model(pluralistic, categorical_columns=cat_cols)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        results[name] = metric_pack(y_test, pred)
        joblib.dump(model, ARTIFACT_DIR / f"{name}_{target}.joblib")
        test_out = test_df[["image_id", "text_prompt", "topic", target]].copy()
        test_out["prediction"] = pred
        test_out["model"] = name
        test_out.to_csv(ARTIFACT_DIR / f"predictions_{name}_{target}.csv", index=False)

        # Group errors are reported only for adequately represented groups.
        if pluralistic:
            group_rows = []
            for col in available_demographics(test_df):
                for value, idx in test_df.groupby(col).groups.items():
                    if len(idx) < MIN_GROUP_SIZE:
                        continue
                    positions = test_df.index.get_indexer(idx)
                    group_rows.append({
                        "attribute": col, "group": str(value), "n": int(len(idx)),
                        "mae": float(mean_absolute_error(y_test.iloc[positions], pred[positions]))
                    })
            pd.DataFrame(group_rows).to_csv(ARTIFACT_DIR / f"group_metrics_{target}.csv", index=False)
    payload = {
        "target": target, "split": "GroupShuffleSplit by image_id",
        "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
        "train_unique_images": int(train_df.image_id.nunique()),
        "test_unique_images": int(test_df.image_id.nunique()),
        "models": results,
    }
    (ARTIFACT_DIR / f"metrics_{target}.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=TARGET_DEFAULT)
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    train(args.target, args.data)
