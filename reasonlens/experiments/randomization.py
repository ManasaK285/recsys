"""
Deterministic, reproducible randomization for ReasonLens.

The same hashing scheme used here is mirrored in
backend/app/routes/experiment.py::deterministic_condition so that
simulated data and live pilot assignments follow identical logic,
letting both feed the same downstream analysis pipeline.
"""
import hashlib
from typing import Tuple

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app.config import APP_TYPES, REASON_TYPES  # noqa: E402


def assign_condition(participant_id: str, seed: int) -> Tuple[str, str]:
    digest = hashlib.sha256(f"{participant_id}:{seed}".encode()).hexdigest()
    app_idx = int(digest[:8], 16) % len(APP_TYPES)
    reason_idx = int(digest[8:16], 16) % len(REASON_TYPES)
    return APP_TYPES[app_idx], REASON_TYPES[reason_idx]


def condition_label(app_type: str, reason_type: str) -> str:
    return f"{app_type}__{reason_type}"
