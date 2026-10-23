import numpy as np
import pandas as pd
from preferenceforge.config import FEATURES

def pairwise_features(df):
    return np.column_stack([
        df[f"a_{f}"].to_numpy() - df[f"b_{f}"].to_numpy()
        for f in FEATURES
    ])

def response_features_from_row(row, side):
    return np.array([row[f"{side}_{f}"] for f in FEATURES], dtype=float)
