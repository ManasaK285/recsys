import numpy as np
from src.models.global_reward import GlobalReward

class IndependentUserReward:
    """One linear reward model per user, with global fallback for sparse users."""
    def __init__(self, feature_dim, lr=.03, epochs=300, l2=1e-3, seed=42):
        self.args = dict(feature_dim=feature_dim, lr=lr, epochs=epochs, l2=l2, seed=seed)
        self.models = {}
        self.global_model = GlobalReward(**self.args)

    def fit(self, a, b, y, user_ids):
        self.global_model.fit(a,b,y)
        for uid in np.unique(user_ids):
            m = user_ids == uid
            if m.sum() >= 3:
                self.models[int(uid)] = GlobalReward(**self.args).fit(a[m],b[m],y[m])
        return self

    def predict_proba(self, a, b, user_ids):
        out = []
        for i, uid in enumerate(user_ids):
            model = self.models.get(int(uid), self.global_model)
            out.append(model.predict_proba(a[i:i+1],b[i:i+1])[0])
        return np.asarray(out)
