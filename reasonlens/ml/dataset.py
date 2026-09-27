"""
Builds a labeled dataset of permission-explanation texts ("reason_text")
with quality attributes: specificity, necessity, transparency,
advertising_disclosure, privacy_language. In simulation mode these are
generated from templates per reason_type with randomized attribute noise;
in pilot mode this file would instead be populated from the actual reason
strings shown by the Android app (recorded in `assignments`/experiment
config), scored by the same attribute rubric.
"""
import numpy as np
import pandas as pd

TEMPLATES = {
    "none": [
        "Allow location access?",
        "This app wants to use your location.",
        "Enable location for this app?",
    ],
    "vague": [
        "We use your location to improve your experience.",
        "Location helps us make the app better for you.",
        "This app uses location data for various features.",
    ],
    "advertising": [
        "Sharing your location lets us show you more relevant ads and offers nearby.",
        "We use location to personalize advertising and recommend nearby deals.",
        "Your location helps our partners deliver targeted promotions.",
    ],
    "functional": [
        "We need your location to show your ride's estimated arrival time.",
        "Location is used to find restaurants near you and estimate delivery time.",
        "We use your location to show local news relevant to your area.",
    ],
    "privacy_preserving": [
        "We use an approximate location, refreshed hourly, only while the app is open, "
        "and never share it with third parties.",
        "Your precise location is never stored; only a coarse area is used and deleted after this session.",
        "Location is processed on your device when possible and is not shared with advertisers.",
    ],
}

# Approximate ground-truth quality attributes per reason_type (0-1 scale),
# used to generate noisy labels for the supervised model to learn.
BASE_ATTRIBUTES = {
    "none": dict(specificity=0.05, necessity=0.10, transparency=0.10, advertising_disclosure=0.0, privacy_language=0.0),
    "vague": dict(specificity=0.20, necessity=0.25, transparency=0.25, advertising_disclosure=0.05, privacy_language=0.05),
    "advertising": dict(specificity=0.55, necessity=0.30, transparency=0.60, advertising_disclosure=0.90, privacy_language=0.05),
    "functional": dict(specificity=0.85, necessity=0.85, transparency=0.70, advertising_disclosure=0.05, privacy_language=0.15),
    "privacy_preserving": dict(specificity=0.75, necessity=0.70, transparency=0.90, advertising_disclosure=0.0, privacy_language=0.95),
}


def build_dataset(n_per_type: int = 150, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for reason_type, templates in TEMPLATES.items():
        base = BASE_ATTRIBUTES[reason_type]
        for _ in range(n_per_type):
            text = templates[rng.integers(0, len(templates))]
            row = {"reason_type": reason_type, "reason_text": text}
            for attr, val in base.items():
                noisy = float(np.clip(rng.normal(val, 0.08), 0, 1))
                row[attr] = noisy
            # overall quality score: weighted composite, held out as the regression target
            row["quality_score"] = float(
                np.clip(
                    0.3 * row["specificity"]
                    + 0.3 * row["necessity"]
                    + 0.25 * row["transparency"]
                    + 0.15 * row["privacy_language"]
                    - 0.2 * row["advertising_disclosure"] * (reason_type == "advertising")
                    + rng.normal(0, 0.03),
                    0,
                    1,
                )
            )
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = build_dataset()
    print(df.head())
    print(df.shape)
