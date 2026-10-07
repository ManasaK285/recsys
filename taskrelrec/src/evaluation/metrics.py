import numpy as np

def recall_at_k(scores,positives,k=10):
    return float(len(set(np.argsort(-scores)[:k]) & set(positives))>0)

def ndcg_at_k(scores,positives,k=10):
    order=np.argsort(-scores)[:k]; pos=set(positives)
    dcg=sum((1 if x in pos else 0)/np.log2(r+2) for r,x in enumerate(order))
    ih=min(len(positives),k)
    if not ih: return 0.0
    idcg=sum(1/np.log2(r+2) for r in range(ih))
    return float(dcg/idcg)

def mrr_at_k(scores,positives,k=10):
    pos=set(positives)
    for r,x in enumerate(np.argsort(-scores)[:k],1):
        if x in pos: return 1/r
    return 0.0
