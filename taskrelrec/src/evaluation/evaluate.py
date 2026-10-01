from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import yaml

from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader

from src.data.dataset import InteractionDataset, TASKS
from src.models.shared_bottom import SharedBottom
from src.models.mmoe import MMoE
from src.models.task_relationship import TaskRelRec
from src.evaluation.metrics import (
    recall_at_k,
    ndcg_at_k,
    mrr_at_k,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"

RANK_NEGATIVES = 99
K = 10
SEED = 42


# ============================================================
# CONFIG
# ============================================================

def load_cfg():
    with open(
        ROOT / "configs/taskrelrec.yaml",
        encoding="utf-8"
    ) as f:
        return yaml.safe_load(f)


# ============================================================
# GLOBAL DIMENSIONS
# ============================================================

def get_global_dimensions():

    data_path = (
        ROOT
        / "data"
        / "processed"
        / "interactions.csv"
    )

    df = pd.read_csv(data_path)

    n_users = int(df["user_id"].max()) + 1
    n_items = int(df["item_id"].max()) + 1

    return n_users, n_items


# ============================================================
# MODEL CONSTRUCTION
# ============================================================

def build_model(
    name,
    n_users,
    n_items,
    config
):

    model_cfg = config["model"]

    common_kwargs = {
        "n_users": n_users,
        "n_items": n_items,
        "n_tasks": len(TASKS),
        "emb_dim": model_cfg["embedding_dim"],
        "hidden": model_cfg["hidden_dim"],
        "dropout": model_cfg["dropout"],
    }

    if name == "shared_bottom":

        model = SharedBottom(
            **common_kwargs
        )

    elif name == "mmoe":

        model = MMoE(
            **common_kwargs
        )

    elif name in [
        "taskrelrec",
        "taskrelrec_no_relation_loss",
    ]:

        model = TaskRelRec(
            **common_kwargs,
            temperature=model_cfg["temperature"],
        )

    else:

        raise ValueError(
            f"Unknown model: {name}"
        )

    return model


# ============================================================
# CHECKPOINT
# ============================================================

def load_model(
    name,
    n_users,
    n_items,
    config
):

    model = build_model(
        name,
        n_users,
        n_items,
        config,
    )

    checkpoint = ART / f"{name}.pt"

    if not checkpoint.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model


# ============================================================
# POINTWISE EVALUATION
# ============================================================

def pointwise_metrics(
    model,
    df,
):

    dataset = InteractionDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
    )

    all_y = []
    all_predictions = []

    with torch.no_grad():

        for users, items, y in loader:

            logits = model(
                users,
                items,
            )

            probabilities = torch.sigmoid(
                logits
            )

            all_y.append(
                y.numpy()
            )

            all_predictions.append(
                probabilities.numpy()
            )

    y_true = np.concatenate(
        all_y,
        axis=0,
    )

    y_pred = np.concatenate(
        all_predictions,
        axis=0,
    )

    results = []

    for task_idx, task in enumerate(TASKS):

        auc = roc_auc_score(
            y_true[:, task_idx],
            y_pred[:, task_idx],
        )

        loss = log_loss(
            y_true[:, task_idx],
            y_pred[:, task_idx],
            labels=[0, 1],
        )

        results.append(
            {
                "task": task,
                "auc": float(auc),
                "logloss": float(loss),
            }
        )

    return results


# ============================================================
# POSITIVE / NEGATIVE CANDIDATE GENERATION
# ============================================================

def build_candidate_sets(
    test_df,
    train_df,
    n_items,
    negatives_per_positive=99,
    seed=42,
):

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Items seen by each user during training.
    #
    # These MUST NOT be used as negative candidates.
    # --------------------------------------------------------

    train_seen = (
        train_df
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    candidates = {}

    test_users = sorted(
        test_df["user_id"].unique()
    )

    for user_id in test_users:

        user_test = test_df[
            test_df["user_id"] == user_id
        ]

        seen_items = train_seen.get(
            user_id,
            set(),
        )

        # ----------------------------------------------------
        # Positive item definition:
        #
        # An item is positive for a task when that task's
        # label is 1 in the held-out test interaction.
        # ----------------------------------------------------

        for task_idx, task in enumerate(TASKS):

            positives = (
                user_test.loc[
                    user_test[task] == 1,
                    "item_id"
                ]
                .astype(int)
                .unique()
                .tolist()
            )

            if len(positives) == 0:
                continue

            # ------------------------------------------------
            # Candidate negatives:
            #
            # 1. Never used in training by this user.
            # 2. Not a positive item for this task.
            # ------------------------------------------------

            positive_set = set(positives)

            forbidden = (
                seen_items
                | positive_set
            )

            available = np.array(
                [
                    item
                    for item in range(n_items)
                    if item not in forbidden
                ],
                dtype=np.int64,
            )

            if len(available) == 0:
                continue

            n_needed = (
                negatives_per_positive
                * len(positives)
            )

            replace = (
                len(available) < n_needed
            )

            negatives = rng.choice(
                available,
                size=n_needed,
                replace=replace,
            )

            candidates[
                (int(user_id), task)
            ] = {
                "positives": np.asarray(
                    positives,
                    dtype=np.int64,
                ),
                "negatives": np.asarray(
                    negatives,
                    dtype=np.int64,
                ),
            }

    return candidates


# ============================================================
# HIT RATE
# ============================================================

def hit_rate_at_k(
    scores,
    positive_mask,
    k=10,
):

    order = np.argsort(
        -scores
    )

    top_k = order[:k]

    return float(
        np.any(
            positive_mask[top_k]
        )
    )


# ============================================================
# PROPER RECOMMENDATION EVALUATION
# ============================================================

def ranking_metrics(
    model,
    test_df,
    train_df,
    n_items,
    k=10,
    negatives_per_positive=99,
    seed=42,
):

    """
    Proper recommendation evaluation.

    For every test user/task:

        positive test items
                 +
        unseen negative items

                 ↓

             candidate set

                 ↓

            model scoring

                 ↓

              ranking

                 ↓

        HR@K
        Recall@K
        NDCG@K
        MRR@K

    Training-seen items are excluded from the negative pool.
    """

    candidate_sets = build_candidate_sets(
        test_df=test_df,
        train_df=train_df,
        n_items=n_items,
        negatives_per_positive=negatives_per_positive,
        seed=seed,
    )

    results = []

    # --------------------------------------------------------
    # Evaluate each user/task separately.
    # --------------------------------------------------------

    for task_idx, task in enumerate(TASKS):

        hit_values = []
        recall_values = []
        ndcg_values = []
        mrr_values = []

        for (
            user_id,
            candidate_info
        ) in candidate_sets.items():

            candidate_user, candidate_task = user_id

            if candidate_task != task:
                continue

            positives = candidate_info[
                "positives"
            ]

            negatives = candidate_info[
                "negatives"
            ]

            candidates = np.concatenate(
                [
                    positives,
                    negatives,
                ]
            )

            labels = np.zeros(
                len(candidates),
                dtype=np.int64,
            )

            labels[
                :len(positives)
            ] = 1
            if isinstance(user_id, tuple):
                user_id = user_id[0]

            user_id = int(user_id)
            users = torch.full(
                (
                    len(candidates),
                ),
                int(user_id),
                dtype=torch.long,
            )

            items = torch.tensor(
                candidates,
                dtype=torch.long,
            )

            with torch.no_grad():

                logits = model(
                    users,
                    items,
                )

                probabilities = (
                    torch.sigmoid(
                        logits
                    )
                    .cpu()
                    .numpy()
                )

            scores = probabilities[
                :,
                task_idx
            ]

            positive_positions = np.where(
                labels == 1
            )[0]

            positive_mask = (
                labels == 1
            )

            hit_values.append(
                hit_rate_at_k(
                    scores,
                    positive_mask,
                    k,
                )
            )

            recall_values.append(
                recall_at_k(
                    scores,
                    positive_positions,
                    k,
                )
            )

            ndcg_values.append(
                ndcg_at_k(
                    scores,
                    positive_positions,
                    k,
                )
            )

            mrr_values.append(
                mrr_at_k(
                    scores,
                    positive_positions,
                    k,
                )
            )

        if hit_values:

            results.append(
                {
                    "task": task,
                    "hitrate@10": float(
                        np.mean(
                            hit_values
                        )
                    ),
                    "recall@10": float(
                        np.mean(
                            recall_values
                        )
                    ),
                    "ndcg@10": float(
                        np.mean(
                            ndcg_values
                        )
                    ),
                    "mrr@10": float(
                        np.mean(
                            mrr_values
                        )
                    ),
                    "users_evaluated": len(
                        hit_values
                    ),
                }
            )

    return results


# ============================================================
# EMPIRICAL TASK RELATIONSHIP
# ============================================================

def empirical_relationship_matrix(
    df,
):

    task_data = (
        df[TASKS]
        .astype(float)
    )

    relationship = (
        task_data
        .corr()
        .fillna(0.0)
    )

    return relationship


def save_empirical_relationship(
    train_df,
):

    relationship = (
        empirical_relationship_matrix(
            train_df
        )
    )

    output_dir = (
        ART / "relationship_matrices"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        output_dir
        / "empirical.csv"
    )

    relationship.to_csv(
        output
    )

    print(
        f"Saved empirical relationship matrix: "
        f"{output}"
    )

    return relationship


# ============================================================
# LEARNED RELATIONSHIP
# ============================================================

def save_relationships(
    model,
    name,
):

    if not hasattr(
        model,
        "relationship_matrix",
    ):
        return None

    relationship = (
        model.relationship_matrix()
    )

    if relationship is None:
        return None

    relationship = (
        relationship
        .detach()
        .cpu()
        .numpy()
    )

    df = pd.DataFrame(
        relationship,
        index=TASKS,
        columns=TASKS,
    )

    output_dir = (
        ART / "relationship_matrices"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        output_dir
        / f"{name}.csv"
    )

    df.to_csv(
        output
    )

    print(
        f"Saved relationship matrix: "
        f"{output}"
    )

    return df


# ============================================================
# RELATIONSHIP COMPARISON
# ============================================================

def compare_relationships(
    empirical,
    learned_relationships,
):

    rows = []

    empirical_values = (
        empirical.to_numpy()
        .flatten()
    )

    for model_name, learned in (
        learned_relationships.items()
    ):

        if learned is None:
            continue

        learned_values = (
            learned.to_numpy()
            .flatten()
        )

        correlation = np.corrcoef(
            empirical_values,
            learned_values,
        )[0, 1]

        mae = np.mean(
            np.abs(
                empirical_values
                - learned_values
            )
        )

        rows.append(
            {
                "model": model_name,
                "relationship_corr": float(
                    correlation
                ),
                "relationship_mae": float(
                    mae
                ),
            }
        )

    result = pd.DataFrame(rows)

    output = (
        ART
        / "relationship_comparison.csv"
    )

    result.to_csv(
        output,
        index=False,
    )

    return result


# ============================================================
# PARAMETER COUNT
# ============================================================

def parameter_count(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# TRANSFER ANALYSIS
# ============================================================

def compute_transfer_gain(
    pointwise_df,
):

    rows = []

    pivot = pointwise_df.pivot(
        index="task",
        columns="model",
        values="auc",
    )

    if "shared_bottom" not in pivot.columns:
        return pd.DataFrame()

    baseline = pivot[
        "shared_bottom"
    ]

    for model_name in pivot.columns:

        if model_name == "shared_bottom":
            continue

        for task in pivot.index:

            gain = (
                pivot.loc[
                    task,
                    model_name
                ]
                - baseline.loc[task]
            )

            rows.append(
                {
                    "model": model_name,
                    "task": task,
                    "baseline": "shared_bottom",
                    "auc_gain": float(gain),
                    "positive_transfer": bool(
                        gain > 0
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    config = load_cfg()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --------------------------------------------------------
    # GLOBAL DIMENSIONS
    # --------------------------------------------------------

    n_users, n_items = (
        get_global_dimensions()
    )

    print(
        f"Global model dimensions:"
        f" users={n_users},"
        f" items={n_items}"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_path = (
        ART / "test.csv"
    )

    if not test_path.exists():

        raise FileNotFoundError(
            "artifacts/test.csv not found."
        )

    test_df = pd.read_csv(
        test_path
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_path = (
        ART / "train.csv"
    )

    if not train_path.exists():

        raise FileNotFoundError(
            "artifacts/train.csv not found."
        )

    train_df = pd.read_csv(
        train_path
    )

    print(
        f"Train rows: {len(train_df):,}"
    )

    print(
        f"Test rows: {len(test_df):,}"
    )

    print(
        f"Test users: "
        f"{test_df.user_id.nunique():,}"
    )

    # --------------------------------------------------------
    # EMPIRICAL RELATIONSHIP
    # --------------------------------------------------------

    empirical = (
        save_empirical_relationship(
            train_df
        )
    )

    # --------------------------------------------------------
    # MODELS CURRENTLY AVAILABLE
    #
    # DO NOT RETRAIN THEM.
    # --------------------------------------------------------

    model_names = [
        "shared_bottom",
        "mmoe",
        "taskrelrec_no_relation_loss",
        "taskrelrec",
    ]

    pointwise_results = []
    ranking_results = []
    complexity_results = []

    learned_relationships = {}

    # --------------------------------------------------------
    # MODEL LOOP
    # --------------------------------------------------------

    for model_name in model_names:

        print()
        print(
            "=" * 70
        )
        print(
            f"Evaluating: {model_name}"
        )
        print(
            "=" * 70
        )

        start_time = time.perf_counter()

        model = load_model(
            model_name,
            n_users,
            n_items,
            config,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        total_params, trainable_params = (
            parameter_count(model)
        )

        complexity_results.append(
            {
                "model": model_name,
                "parameters": total_params,
                "trainable_parameters": (
                    trainable_params
                ),
                "checkpoint_load_seconds": (
                    elapsed
                ),
            }
        )

        # ----------------------------------------------------
        # POINTWISE
        # ----------------------------------------------------

        pointwise = pointwise_metrics(
            model,
            test_df,
        )

        for result in pointwise:

            result["model"] = (
                model_name
            )

            pointwise_results.append(
                result
            )

        # ----------------------------------------------------
        # PROPER RANKING
        # ----------------------------------------------------

        print(
            "Running proper "
            "positive + negative "
            "candidate ranking..."
        )

        ranking = ranking_metrics(
            model=model,
            test_df=test_df,
            train_df=train_df,
            n_items=n_items,
            k=K,
            negatives_per_positive=(
                RANK_NEGATIVES
            ),
            seed=SEED,
        )

        for result in ranking:

            result["model"] = (
                model_name
            )

            ranking_results.append(
                result
            )

        # ----------------------------------------------------
        # LEARNED RELATIONSHIP
        # ----------------------------------------------------

        learned = save_relationships(
            model,
            model_name,
        )

        if learned is not None:

            learned_relationships[
                model_name
            ] = learned

    # ========================================================
    # DATAFRAMES
    # ========================================================

    pointwise_df = pd.DataFrame(
        pointwise_results
    )

    pointwise_df = pointwise_df[
        [
            "model",
            "task",
            "auc",
            "logloss",
        ]
    ]

    ranking_df = pd.DataFrame(
        ranking_results
    )

    ranking_df = ranking_df[
        [
            "model",
            "task",
            "hitrate@10",
            "recall@10",
            "ndcg@10",
            "mrr@10",
            "users_evaluated",
        ]
    ]

    complexity_df = pd.DataFrame(
        complexity_results
    )

    # ========================================================
    # SAVE POINTWISE
    # ========================================================

    pointwise_path = (
        ART
        / "pointwise_metrics.csv"
    )

    pointwise_df.to_csv(
        pointwise_path,
        index=False,
    )

    # ========================================================
    # SAVE RANKING
    # ========================================================

    ranking_path = (
        ART
        / "ranking_metrics.csv"
    )

    ranking_df.to_csv(
        ranking_path,
        index=False,
    )

    # ========================================================
    # SAVE COMPLEXITY
    # ========================================================

    complexity_path = (
        ART
        / "model_complexity.csv"
    )

    complexity_df.to_csv(
        complexity_path,
        index=False,
    )

    # ========================================================
    # RELATIONSHIP COMPARISON
    # ========================================================

    if learned_relationships:

        compare_relationships(
            empirical,
            learned_relationships,
        )

    # ========================================================
    # TRANSFER GAIN
    # ========================================================

    transfer_df = (
        compute_transfer_gain(
            pointwise_df
        )
    )

    transfer_path = (
        ART
        / "transfer_gain.csv"
    )

    transfer_df.to_csv(
        transfer_path,
        index=False,
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    final_df = pointwise_df.merge(
        ranking_df,
        on=["model", "task"],
        how="outer",
    )

    final_path = (
        ART
        / "final_results.csv"
    )

    final_df.to_csv(
        final_path,
        index=False,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print(
        "=" * 90
    )
    print(
        "FINAL TEST POINTWISE METRICS"
    )
    print(
        "=" * 90
    )

    print(
        pointwise_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print(
        "=" * 90
    )
    print(
        "FINAL PROPER RANKING METRICS"
    )
    print(
        "=" * 90
    )

    print(
        ranking_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print(
        "=" * 90
    )
    print(
        "MODEL COMPLEXITY"
    )
    print(
        "=" * 90
    )

    print(
        complexity_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print(
        "=" * 90
    )
    print(
        "TASKRELREC V3 ARTIFACTS"
    )
    print(
        "=" * 90
    )

    print(
        f"  {pointwise_path}"
    )

    print(
        f"  {ranking_path}"
    )

    print(
        f"  {complexity_path}"
    )

    print(
        f"  {transfer_path}"
    )

    print(
        f"  {final_path}"
    )

    print(
        f"  {ART / 'relationship_matrices'}"
    )


if __name__ == "__main__":
    main()
