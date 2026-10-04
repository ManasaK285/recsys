from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import Environment
from .models import train_all
from .evaluation import evaluate_models

def split_data(log, seed):
    ids = log["impression_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    cut = int(0.8*len(ids))
    return (
        log[log.impression_id.isin(ids[:cut])].copy(),
        log[log.impression_id.isin(ids[cut:])].copy(),
    )

def run_once(users=700, impressions=24, candidates=20, bias=0.75, seed=42):
    env = Environment(users, candidates, impressions, seed, bias)
    start = time.perf_counter()
    log, _ = env.generate()
    train, test = split_data(log, seed)
    models = train_all(train, seed)
    metrics = evaluate_models(test, models)
    return {
        "metrics": metrics,
        "seconds": time.perf_counter()-start,
        "rows_train": len(train),
        "rows_test": len(test),
        "bias": bias,
        "seed": seed,
    }

def validation_debias_experiment(users=300, seed=43):
    env = Environment(users, 20, 20, seed, 0.85)
    log, _ = env.generate()
    train, rest = split_data(log, seed)
    val_ids = rest["impression_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(val_ids)
    half = len(val_ids)//2
    val = rest[rest.impression_id.isin(val_ids[:half])].copy()

    records = []
    for leaves in [7, 15, 31]:
        model = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.08,
            max_leaf_nodes=leaves, random_state=seed
        )
        cols = ["x0","x1","x2","x3","position"]
        model.fit(train[cols], train["click"])
        val["score"] = model.predict_proba(val[cols])[:,1]

        top = (val.sort_values(["impression_id","score"], ascending=[True,False])
               .groupby("impression_id").head(10))
        biased = top.groupby("impression_id")["click"].max().mean()

        val["debias_click"] = val["click"] / np.maximum(val["exposure"], 0.05)
        top2 = (val.sort_values(["impression_id","score"], ascending=[True,False])
                .groupby("impression_id").head(10))
        debiased = top2.groupby("impression_id")["debias_click"].mean().mean()

        records.append({
            "max_leaf_nodes": leaves,
            "biased_validation": float(biased),
            "debiased_validation": float(debiased),
        })
    return records

def save_report(result, validation, outdir="artifacts"):
    out = Path(outdir)
    out.mkdir(exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(result, indent=2))
    (out/"validation.json").write_text(json.dumps(validation, indent=2))

    rows = [{"method": m, **v} for m,v in result["metrics"].items()]
    df = pd.DataFrame(rows)
    df.to_csv(out/"metrics.csv", index=False)

    ax = df.plot(
        x="method",
        y=["ndcg@10","mrr@10","hit_rate@10","pairwise_accuracy"],
        kind="bar", figsize=(10,6),
        title="Position-Bias Debiasing Comparison"
    )
    ax.figure.tight_layout()
    ax.figure.savefig(out/"model_comparison.png", dpi=150)
    plt.close(ax.figure)

    vdf = pd.DataFrame(validation)
    ax = vdf.plot(
        x="max_leaf_nodes",
        y=["biased_validation","debiased_validation"],
        marker="o", figsize=(9,5),
        title="Validation Objective Under Position Bias"
    )
    ax.figure.tight_layout()
    ax.figure.savefig(out/"validation_bias.png", dpi=150)
    plt.close(ax.figure)
