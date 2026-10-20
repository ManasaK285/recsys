from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import log_loss, accuracy_score
from .data import save_synthetic_data
from .models import (
    q_learning_predict, episodic_memory_predict, sequence_tensor_for_group,
    q_values_before_trials, HistoryGRU, HybridQGRU, neural_parameter_count
)


def score_probs(y, probs):
    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-7, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    onehot = np.eye(probs.shape[1])[y]
    return {
        "log_loss": float(log_loss(y, probs, labels=list(range(probs.shape[1])))),
        "accuracy": float(accuracy_score(y, probs.argmax(axis=1))),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
    }


def _baseline_rows(df, fn, name):
    pred_rows = []
    for pid, trial, y, probs, reward in fn(df):
        pred_rows.append({"participant_id": pid, "trial": trial, "choice": y,
                          "model": name, **{f"p{a}": probs[a] for a in range(4)}})
    return pd.DataFrame(pred_rows)


def train_neural_model(train_df, test_df, kind="gru", epochs=12, seed=42, hidden_dim=32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HistoryGRU(hidden_dim=hidden_dim) if kind == "gru" else HybridQGRU(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_examples = []
    for pid, g in train_df.groupby("participant_id"):
        X, y = sequence_tensor_for_group(g)
        qv = q_values_before_trials(g)
        train_examples.append((torch.tensor(X), torch.tensor(y, dtype=torch.long),
                               torch.tensor(qv, dtype=torch.float32)))
    for _ in range(epochs):
        model.train()
        np.random.shuffle(train_examples)
        for X, y, qv in train_examples:
            optimizer.zero_grad()
            logits = model(X, qv) if kind == "hybrid" else model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
    model.eval()
    records = []
    with torch.no_grad():
        for pid, g in test_df.groupby("participant_id"):
            X, y = sequence_tensor_for_group(g)
            qv = q_values_before_trials(g)
            X_t = torch.tensor(X)
            logits = model(X_t, torch.tensor(qv)) if kind == "hybrid" else model(X_t)
            probs = torch.softmax(logits, dim=-1).numpy()
            for i, (_, row) in enumerate(g.sort_values("trial").iterrows()):
                records.append({"participant_id": int(pid), "trial": int(row["trial"]),
                                "choice": int(y[i]), "model": "hybrid_q_gru" if kind == "hybrid" else "gru_history",
                                **{f"p{a}": float(probs[i, a]) for a in range(4)}})
    return pd.DataFrame(records), neural_parameter_count(model)


def evaluate_all(df, epochs=12, seed=42):
    pids = np.array(sorted(df.participant_id.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n_test = max(1, int(round(0.2 * len(pids))))
    test_ids = set(pids[:n_test].tolist())
    train_df = df[~df.participant_id.isin(test_ids)].copy()
    test_df = df[df.participant_id.isin(test_ids)].copy()

    predictions = [
        _baseline_rows(test_df, lambda d: q_learning_predict(d), "q_learning"),
        _baseline_rows(test_df, lambda d: episodic_memory_predict(d), "episodic_memory"),
    ]
    param_counts = {"q_learning": 2, "episodic_memory": 2}
    for kind, name in [("gru", "gru_history"), ("hybrid", "hybrid_q_gru")]:
        pred, count = train_neural_model(train_df, test_df, kind=kind, epochs=epochs, seed=seed)
        predictions.append(pred)
        param_counts[name] = count

    pred_df = pd.concat(predictions, ignore_index=True)
    metric_rows = []
    for model_name, group in pred_df.groupby("model"):
        group = group.sort_values(["participant_id", "trial"])
        probs = group[[f"p{a}" for a in range(4)]].to_numpy()
        metrics = score_probs(group.choice.to_numpy(), probs)
        metric_rows.append({"model": model_name, **metrics,
                            "parameters": int(param_counts[model_name]),
                            "n_test_trials": len(group)})
    metrics_df = pd.DataFrame(metric_rows).sort_values("log_loss")
    return metrics_df, pred_df, train_df, test_df


def run_experiment(participants=120, trials=150, epochs=12, seed=42):
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    df = save_synthetic_data(n_participants=participants, n_trials=trials, seed=seed)
    metrics, preds, train_df, test_df = evaluate_all(df, epochs=epochs, seed=seed)
    metrics.to_csv("results/metrics.csv", index=False)
    preds.to_csv("results/predictions.csv", index=False)
    summary = {
        "seed": seed, "participants": participants, "trials_per_participant": trials,
        "train_participants": int(train_df.participant_id.nunique()),
        "test_participants": int(test_df.participant_id.nunique()),
        "epochs": epochs, "best_by_log_loss": str(metrics.iloc[0]["model"]),
        "metrics": metrics.to_dict(orient="records"),
        "note": "Synthetic-data experiment; not a reproduction of published human results."
    }
    with open("results/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return metrics, preds


if __name__ == "__main__":
    run_experiment()
