from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Environment:
    users: int = 700
    candidates_per_impression: int = 20
    impressions_per_user: int = 24
    seed: int = 42
    position_bias: float = 0.75

    def generate(self):
        rng = np.random.default_rng(self.seed)
        rows = []
        truth = []

        for user in range(self.users):
            user_vec = rng.normal(size=4)

            for imp in range(self.impressions_per_user):
                n = self.candidates_per_impression
                x = rng.normal(size=(n, 4))
                item_ids = np.arange(n) + (user * self.impressions_per_user + imp) * n

                linear = x @ user_vec / 2.0
                nonlinear = 0.8*x[:,0]*x[:,1] - 0.4*x[:,2]**2 + 0.3*np.sin(x[:,3])
                relevance_logit = linear + nonlinear
                relevance = 1 / (1 + np.exp(-relevance_logit))

                # Endogenous logging policy plus exogenous ranking perturbation.
                base = relevance_logit + rng.normal(0, 0.9, n)
                instrument = rng.normal(0, 1, n)
                order = np.argsort(-(base + 0.45*instrument))

                position = np.empty(n, dtype=int)
                position[order] = np.arange(1, n+1)

                exposure = np.exp(-self.position_bias * (position-1) / 4.0)
                click_logit = relevance_logit + np.log(np.maximum(exposure, 1e-8))
                click_prob = 1 / (1 + np.exp(-click_logit))
                click = rng.binomial(1, click_prob)

                impression_id = user*self.impressions_per_user + imp

                for j in range(n):
                    row = {
                        "user_id": user,
                        "impression_id": impression_id,
                        "item_id": int(item_ids[j]),
                        "position": int(position[j]),
                        "instrument": float(instrument[j]),
                        "click": int(click[j]),
                        "relevance": float(relevance[j]),
                        "exposure": float(exposure[j]),
                    }
                    for k in range(4):
                        row[f"x{k}"] = float(x[j,k])
                    rows.append(row)
                    truth.append({
                        "user_id": user,
                        "impression_id": impression_id,
                        "item_id": int(item_ids[j]),
                        "relevance": float(relevance[j]),
                    })

        return pd.DataFrame(rows), pd.DataFrame(truth)
