from pathlib import Path
import numpy as np
import pandas as pd


def simulate_bandit_data(n_participants=120, n_trials=150, n_actions=4, seed=42):
    """Create synthetic trial sequences in a drifting four-armed bandit task."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_participants):
        # Smoothly drifting reward probabilities, with independent participant schedules.
        probs = rng.uniform(0.25, 0.75, size=n_actions)
        alpha = rng.uniform(0.015, 0.08)
        beta = rng.uniform(2.0, 7.0)
        q = np.full(n_actions, 0.5)
        for t in range(n_trials):
            if t:
                probs += rng.normal(0, 0.035, size=n_actions)
                probs = np.clip(probs, 0.05, 0.95)
            logits = beta * q
            logits -= logits.max()
            soft = np.exp(logits)
            soft /= soft.sum()
            # Some exploration and individual variability.
            if rng.random() < 0.06:
                choice = int(rng.integers(n_actions))
            else:
                choice = int(rng.choice(n_actions, p=soft))
            reward = int(rng.random() < probs[choice])
            q[choice] += alpha * (reward - q[choice])
            row = {
                "participant_id": p,
                "trial": t,
                "choice": choice,
                "reward": reward,
            }
            for a in range(n_actions):
                row[f"reward_{a}"] = float(probs[a])
            rows.append(row)
    return pd.DataFrame(rows)


def save_synthetic_data(out_path="data/processed/synthetic_trials.csv",
                        n_participants=120, n_trials=150, seed=42):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = simulate_bandit_data(n_participants, n_trials, seed=seed)
    df.to_csv(path, index=False)
    return df
