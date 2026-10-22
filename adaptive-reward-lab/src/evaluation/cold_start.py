import numpy as np
from src.evaluation.metrics import evaluate_probabilities

def cold_start_curve(model, data, support_sizes=(1,2,5,10,20), seed=7):
    rng=np.random.default_rng(seed); rows=[]
    for uid in np.unique(data.user_ids):
        idx=np.where(data.user_ids==uid)[0]; rng.shuffle(idx)
        for k in support_sizes:
            if len(idx)<=k: continue
            s,t=idx[:k],idx[k:]
            w=model.adapt_user(data.features_a[s],data.features_b[s],data.labels[s])
            rows.append({"user_id":int(uid),"support_size":int(k),
                         **evaluate_probabilities(data.labels[t],model.predict_with_weights(data.features_a[t],data.features_b[t],w))})
    return rows
