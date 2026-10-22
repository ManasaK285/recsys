import numpy as np

class GlobalReward:
    """Single reward vector shared across users, optimized with pairwise logistic loss."""
    def __init__(self, feature_dim, lr=.03, epochs=300, l2=1e-3, seed=42):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.w = np.random.default_rng(seed).normal(0, .01, feature_dim)

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-np.clip(z, -30, 30)))

    def fit(self, a, b, y):
        x, y = a-b, np.asarray(y, dtype=float)
        for _ in range(self.epochs):
            p = self.sigmoid(x @ self.w)
            self.w -= self.lr * (x.T @ (p-y)/max(len(y),1) + self.l2*self.w)
        return self

    def predict_proba(self, a, b, user_ids=None):
        return self.sigmoid((a-b) @ self.w)
