from pathlib import Path
import pandas as pd
from preferenceforge.config import FEATURES

def append_human_preference(user_id, prompt, response_a, response_b, features_a, features_b, label, path="data/human_preferences.csv"):
    """Append one human comparison. label=1 means A preferred, label=0 means B preferred."""
    if label not in (0, 1):
        raise ValueError("label must be 0 (B) or 1 (A)")
    row = {"user_id": user_id, "prompt": prompt, "response_a": response_a,
           "response_b": response_b, "label": int(label)}
    for f in FEATURES:
        row[f"a_{f}"] = float(features_a[f])
        row[f"b_{f}"] = float(features_b[f])
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(p, mode="a", header=not p.exists(), index=False)
    return p
