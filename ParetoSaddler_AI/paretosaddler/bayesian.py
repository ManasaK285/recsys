import numpy as np
import pandas as pd

def _task_rollouts(frame: pd.DataFrame):
    grouped = {}
    for task_id, group in frame.groupby("task_id"):
        grouped[task_id] = (
            group["quality"].to_numpy(dtype=float),
            group["cost"].to_numpy(dtype=float),
        )
    return grouped

def bayesian_promotion_support(parent: pd.DataFrame, candidate: pd.DataFrame,
                               draws: int = 3000, gamma: float = 0.70,
                               quality_loss: float = 0.10,
                               cost_increase: float = float("inf"),
                               seed: int = 42):
    """
    Bayesian bootstrap with independent Dirichlet weights per task and harness.
    Cost movement is relative to parent mean cost. Quality is an absolute delta.
    """
    p = _task_rollouts(parent)
    c = _task_rollouts(candidate)
    common = sorted(set(p).intersection(c))
    if not common:
        raise ValueError("Parent and candidate must share screening task IDs.")
    rng = np.random.default_rng(seed)
    support = np.zeros(draws, dtype=bool)

    for j in range(draws):
        pq, pc, cq, cc = [], [], [], []
        for task_id in common:
            p_quality, p_cost = p[task_id]
            c_quality, c_cost = c[task_id]
            wp = rng.dirichlet(np.ones(len(p_quality)))
            wc = rng.dirichlet(np.ones(len(c_quality)))
            pq.append(float(wp @ p_quality))
            pc.append(float(wp @ p_cost))
            cq.append(float(wc @ c_quality))
            cc.append(float(wc @ c_cost))
        delta_q = float(np.mean(cq) - np.mean(pq))
        parent_cost = max(float(np.mean(pc)), 1e-12)
        delta_c = float((np.mean(cc) - np.mean(pc)) / parent_cost)
        quality_gain = (delta_q > 0) and (delta_c <= cost_increase)
        cost_gain = (delta_c < 0) and (delta_q >= -quality_loss)
        support[j] = quality_gain or cost_gain

    score = float(support.mean())
    return {"support": score, "promoted": bool(score >= gamma),
            "draws": int(draws), "gamma": float(gamma)}
