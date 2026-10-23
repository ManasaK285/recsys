from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from preferenceforge.models import GlobalRewardModel, IndependentRewardModel, AdaptiveRewardFeatureModel, SupportSummaryModel

def metrics_for(y, p):
    pred = (p >= 0.5).astype(int)
    p = np.clip(p, 1e-6, 1-1e-6)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, p, labels=[0,1])),
        "brier": float(brier_score_loss(y, p)),
        "n": int(len(y))
    }

def evaluate(data_dir="artifacts"):
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    models = {
        "global": GlobalRewardModel().fit(train),
        "independent": IndependentRewardModel().fit(train),
        "adaptive_reward_features": AdaptiveRewardFeatureModel().fit(train, support_size=8),
        "support_summary": SupportSummaryModel().fit(train, support_size=8),
    }
    rows, pred_rows = [], []
    y = test["label"].astype(int).to_numpy()
    for name, model in models.items():
        p = model.predict_proba(test)
        m = metrics_for(y, p); m["model"] = name; rows.append(m)
        pred_rows.extend({"index": int(idx), "model": name, "y_true": int(yy), "p_prefer_a": float(pp)}
                         for idx, yy, pp in zip(test.index, y, p))
    metrics = pd.DataFrame(rows)[["model", "accuracy", "log_loss", "brier", "n"]]
    Path(data_dir).mkdir(exist_ok=True)
    metrics.to_csv(f"{data_dir}/metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(f"{data_dir}/predictions.csv", index=False)
    Path(f"{data_dir}/metrics.json").write_text(metrics.to_json(orient="records", indent=2), encoding="utf-8")
    return metrics
