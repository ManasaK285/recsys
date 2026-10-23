import numpy as np
from sklearn.linear_model import LogisticRegression
from preferenceforge.features import pairwise_features
from preferenceforge.config import FEATURES

class GlobalRewardModel:
    """One shared reward vector for all raters."""
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
    def fit(self, df):
        self.model.fit(pairwise_features(df), df["label"].astype(int))
        return self
    def predict_proba(self, df):
        return self.model.predict_proba(pairwise_features(df))[:, 1]
    def export(self):
        return {"weights": self.model.coef_[0].tolist(), "intercept": float(self.model.intercept_[0])}

class IndependentRewardModel:
    """Separate linear preference model per user; needs enough user labels."""
    def __init__(self):
        self.models = {}
        self.fallback = GlobalRewardModel()
    def fit(self, df):
        self.fallback.fit(df)
        for uid, part in df.groupby("user_id"):
            if part["label"].nunique() < 2 or len(part) < 4:
                continue
            model = LogisticRegression(max_iter=1000)
            model.fit(pairwise_features(part), part["label"].astype(int))
            self.models[uid] = model
        return self
    def predict_proba(self, df):
        out = np.zeros(len(df))
        for uid, indices in df.groupby("user_id").groups.items():
            idx = np.array(list(indices))
            part = df.loc[idx]
            model = self.models.get(uid)
            out[np.searchsorted(df.index.to_numpy(), idx)] = (
                model.predict_proba(pairwise_features(part))[:, 1]
                if model else self.fallback.predict_proba(part)
            )
        return out
    def export(self):
        return {uid: model.coef_[0].tolist() for uid, model in self.models.items()}

class AdaptiveRewardFeatureModel:
    """
    Shared feature representation + user-specific weights estimated from a small support set.
    Here features are explicit, interpretable response attributes; the shared representation
    is identity and shared feature importance is learned with logistic regression.
    """
    def __init__(self, ridge=2.0):
        self.ridge = ridge
        self.global_model = GlobalRewardModel()
        self.user_weights = {}
    def fit(self, train_df, support_size=8):
        self.global_model.fit(train_df)
        self.user_weights = {}
        for uid, part in train_df.groupby("user_id"):
            support = part.head(support_size)
            # MAP-style shrinkage: combine a small user's fitted direction with global weights.
            global_w = self.global_model.model.coef_[0]
            if support["label"].nunique() < 2 or len(support) < 3:
                self.user_weights[uid] = global_w.copy()
                continue
            X = pairwise_features(support)
            y = support["label"].astype(int).to_numpy()
            # Gradient descent on logistic loss + L2 shrinkage toward shared weights.
            w = global_w.copy()
            for _ in range(250):
                p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
                grad = X.T @ (p - y) / len(y) + self.ridge * (w - global_w) / len(w)
                w -= 0.08 * grad
            self.user_weights[uid] = w
        return self
    def predict_proba(self, df):
        X = pairwise_features(df)
        out = np.zeros(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            w = self.user_weights.get(row["user_id"], self.global_model.model.coef_[0])
            z = float(X[i] @ w + self.global_model.model.intercept_[0])
            out[i] = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return out
    def export(self):
        return {uid: np.asarray(w).tolist() for uid, w in self.user_weights.items()}

class SupportSummaryModel:
    """Simple support-set-only baseline; no cross-user shared feature learning."""
    def __init__(self):
        self.weights = {}
        self.global_model = GlobalRewardModel()
    def fit(self, train_df, support_size=8):
        self.global_model.fit(train_df)
        for uid, part in train_df.groupby("user_id"):
            support = part.head(support_size)
            X = pairwise_features(support)
            y = support["label"].to_numpy()
            w = X.T @ (2*y - 1) / max(1, len(y))
            self.weights[uid] = w
        return self
    def predict_proba(self, df):
        X = pairwise_features(df)
        out = []
        for i, (_, row) in enumerate(df.iterrows()):
            w = self.weights.get(row["user_id"], self.global_model.model.coef_[0])
            z = float(X[i] @ w)
            out.append(1 / (1 + np.exp(-np.clip(z, -30, 30))))
        return np.asarray(out)
    def export(self):
        return {uid: np.asarray(w).tolist() for uid, w in self.weights.items()}
