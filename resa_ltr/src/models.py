import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge

FEATURES = ["x0", "x1", "x2", "x3"]

class NaiveLTR:
    def __init__(self, seed=42):
        self.model = HistGradientBoostingClassifier(
            max_iter=140, learning_rate=0.08, max_leaf_nodes=15,
            random_state=seed
        )

    def fit(self, df):
        self.model.fit(df[FEATURES + ["position"]], df["click"])
        return self

    def score(self, df):
        return self.model.predict_proba(df[FEATURES + ["position"]])[:,1]

class IPSLTR:
    def __init__(self, seed=42):
        self.model = HistGradientBoostingClassifier(
            max_iter=140, learning_rate=0.08, max_leaf_nodes=15,
            random_state=seed
        )

    def fit(self, df):
        props = df["position"].value_counts(normalize=True)
        p = df["position"].map(props).to_numpy()
        weights = np.clip(1 / np.maximum(p, 0.02), 0, 30)
        self.model.fit(
            df[FEATURES + ["position"]], df["click"], sample_weight=weights
        )
        return self

    def score(self, df):
        return self.model.predict_proba(df[FEATURES + ["position"]])[:,1]

class ControlFunctionLTR:
    """
    Lightweight control-function implementation.

    Stage 1 predicts logged position using item features + an exogenous
    ranking perturbation. The residual is carried into Stage 2.
    Ranking scores are evaluated at a common top position.
    """
    def __init__(self, seed=42):
        self.position_model = Ridge(alpha=1.0)
        self.click_model = HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.07, max_leaf_nodes=15,
            random_state=seed
        )

    def fit(self, df):
        X1 = df[FEATURES + ["instrument"]]
        y1 = df["position"].astype(float)
        self.position_model.fit(X1, y1)

        residual = y1.to_numpy() - self.position_model.predict(X1)
        train = df.copy()
        train["control_residual"] = residual

        X2 = train[FEATURES + ["position", "control_residual"]]
        self.click_model.fit(X2, train["click"])
        return self

    def score(self, df):
        frame = df.copy()
        predicted_position = self.position_model.predict(
            frame[FEATURES + ["instrument"]]
        )
        frame["control_residual"] = (
            frame["position"].to_numpy() - predicted_position
        )

        # Counterfactual comparison: every candidate gets equal top exposure.
        frame["position"] = 1
        return self.click_model.predict_proba(
            frame[FEATURES + ["position", "control_residual"]]
        )[:,1]

class TrainedModels:
    def __init__(self, naive, ips, control):
        self.naive = naive
        self.ips = ips
        self.control = control

def train_all(train, seed=42):
    return TrainedModels(
        NaiveLTR(seed).fit(train),
        IPSLTR(seed).fit(train),
        ControlFunctionLTR(seed).fit(train),
    )
