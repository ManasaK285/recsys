"""
Generates reproducible synthetic ReasonLens data for development, testing,
and pipeline validation ("simulation mode"). This is clearly separated from
real pilot data collected through the Android app + backend: simulated rows
are always tagged mode="simulation" so the analysis layer (and anyone
reading data/synthetic/) can never confuse the two.
"""
import os
import uuid
import yaml
import numpy as np
import pandas as pd

from randomization import assign_condition, condition_label

HERE = os.path.dirname(__file__)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_config(path=None):
    path = path or os.path.join(HERE, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def generate(cfg: dict, seed: int = None):
    seed = seed if seed is not None else cfg["seed"]
    rng = np.random.default_rng(seed)
    sim_cfg = cfg["simulation"]
    n = sim_cfg["n_participants"]

    participants, assignments, events, surveys = [], [], [], []

    demographic_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]

    for i in range(n):
        participant_id = f"sim-{seed}-{i:06d}-{uuid.uuid4().hex[:8]}"
        demo = demographic_groups[rng.integers(0, len(demographic_groups))]

        app_type, reason_type = assign_condition(participant_id, cfg["seed"])
        condition = condition_label(app_type, reason_type)

        participants.append(
            {
                "participant_id": participant_id,
                "demographic_group": demo,
                "mode": "simulation",
            }
        )
        assignments.append(
            {
                "participant_id": participant_id,
                "experiment_id": cfg["experiment_id"],
                "app_type": app_type,
                "reason_type": reason_type,
                "condition": condition,
                "seed": cfg["seed"],
            }
        )

        # --- behavioral model: permission decision ---
        logit = _sigmoid_input(sim_cfg["base_grant_rate"])
        logit += sim_cfg["reason_type_grant_effect"][reason_type]
        logit += sim_cfg["app_type_grant_effect"][app_type]
        grant_prob = float(np.clip(_sigmoid(logit), 0.02, 0.98))
        decision = "granted" if rng.random() < grant_prob else "denied"

        precision = None
        if decision == "granted":
            precise_logit = _sigmoid_input(sim_cfg["base_precise_rate"])
            precise_logit += 0.5 * sim_cfg["reason_type_grant_effect"][reason_type]
            precise_prob = float(np.clip(_sigmoid(precise_logit), 0.02, 0.98))
            precision = "precise" if rng.random() < precise_prob else "approximate"

        rt = max(
            300,
            int(rng.normal(sim_cfg["response_time_ms_mean"], sim_cfg["response_time_ms_sd"])),
        )

        events.append(
            {
                "participant_id": participant_id,
                "event": "permission_result",
                "decision": decision,
                "precision": precision,
                "response_time_ms": rt,
            }
        )

        # --- behavioral model: trust survey ---
        trust_base = 3.0 + sim_cfg["reason_type_trust_effect"][reason_type]
        app_trust = _bounded_likert(rng, trust_base)
        android_trust = _bounded_likert(rng, 3.0 + 0.3 * sim_cfg["reason_type_trust_effect"][reason_type])
        necessity = _bounded_likert(rng, 3.0 + 0.6 * sim_cfg["reason_type_grant_effect"][reason_type] * 5)
        privacy_concern = _bounded_likert(rng, 3.0 - 0.5 * sim_cfg["reason_type_trust_effect"][reason_type])

        surveys.append(
            {
                "participant_id": participant_id,
                "app_trust": app_trust,
                "android_trust": android_trust,
                "necessity": necessity,
                "privacy_concern": privacy_concern,
            }
        )

    return (
        pd.DataFrame(participants),
        pd.DataFrame(assignments),
        pd.DataFrame(events),
        pd.DataFrame(surveys),
    )


def _sigmoid_input(rate: float) -> float:
    """Convert a target base rate into the logit that produces it."""
    rate = float(np.clip(rate, 1e-6, 1 - 1e-6))
    return float(np.log(rate / (1 - rate)))


def _bounded_likert(rng, center: float) -> int:
    val = int(round(rng.normal(center, 0.8)))
    return int(np.clip(val, 1, 5))


def run(config_path=None, out_dir=None):
    cfg = load_config(config_path)
    out_dir = out_dir or os.path.join(HERE, "..", cfg["output"]["synthetic_dir"])
    os.makedirs(out_dir, exist_ok=True)

    participants, assignments, events, surveys = generate(cfg)

    participants.to_csv(os.path.join(out_dir, "participants.csv"), index=False)
    assignments.to_csv(os.path.join(out_dir, "assignments.csv"), index=False)
    events.to_csv(os.path.join(out_dir, "permission_events.csv"), index=False)
    surveys.to_csv(os.path.join(out_dir, "survey_responses.csv"), index=False)

    print(f"[simulation] wrote {len(participants)} participants to {out_dir}")
    return participants, assignments, events, surveys


if __name__ == "__main__":
    run()
