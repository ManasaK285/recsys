"""
Central configuration for ReasonLens backend.

APP_TYPES and REASON_TYPES here are the single source of truth and are
imported by the experiment randomization engine, the simulation module,
and the analysis layer so that all components always agree on the set
of experimental conditions.
"""
import os
from typing import List

APP_TYPES: List[str] = [
    "rideshare",
    "food_delivery",
    "local_news",
    "wallpaper",
]

REASON_TYPES: List[str] = [
    "none",
    "functional",
    "vague",
    "advertising",
    "privacy_preserving",
]

PRECISION_LEVELS: List[str] = ["approximate", "precise"]
DECISIONS: List[str] = ["granted", "denied"]

# "simulation" -> synthetic, reproducible data for development/testing.
# "pilot"      -> real participant data collected via the Android app,
#                 gated on the consent screen recorded in `participants`.
MODES: List[str] = ["simulation", "pilot"]

DATABASE_URL = os.environ.get(
    "REASONLENS_DATABASE_URL",
    f"sqlite:///{os.path.join(os.path.dirname(__file__), '..', 'reasonlens.db')}",
)

DEFAULT_SEED = int(os.environ.get("REASONLENS_SEED", "42"))
