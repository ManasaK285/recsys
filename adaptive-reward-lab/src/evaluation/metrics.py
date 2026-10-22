import numpy as np

def evaluate_probabilities(y_true, probabilities):
    y=np.asarray(y_true,dtype=float); p=np.clip(np.asarray(probabilities,dtype=float),1e-7,1-1e-7)
    return {"accuracy":float(np.mean((p>=.5)==y)),
            "log_loss":float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
            "brier":float(np.mean((p-y)**2)),"n":int(len(y))}
