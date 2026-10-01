import numpy as np
TASK_WEIGHTS=np.array([.15,.20,.25,.40],dtype=np.float32)
def rank_items(probabilities,item_ids,top_k=10):
    scores=np.asarray(probabilities)@TASK_WEIGHTS
    order=np.argsort(-scores)[:top_k]
    return [{"item_id":int(item_ids[i]),"score":float(scores[i])} for i in order]
