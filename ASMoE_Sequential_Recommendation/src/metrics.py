import numpy as np, torch

def rank_target(scores,target,seen):
    s=scores.clone()
    if seen: s[list(seen)]=-float("inf")
    rank=int((s>s[target]).sum().item())+1
    return rank

@torch.no_grad()
def evaluate_model(model,examples,num_items,ks=(5,10,20),device="cpu"):
    model.eval(); vals={f"Recall@{k}":[] for k in ks}|{f"NDCG@{k}":[] for k in ks}|{"MRR":[]}
    for ex in examples:
        x=torch.tensor([ex["history"]],dtype=torch.long,device=device)
        c=torch.tensor([ex["history_categories"]],dtype=torch.long,device=device)
        valid=torch.ones_like(x,dtype=torch.bool)
        scores,_=model(x,c,valid); scores=scores[0].cpu(); rank=rank_target(scores,int(ex["target"]),set(ex["history"]))
        vals["MRR"].append(1/rank)
        for k in ks:
            hit=rank<=k; vals[f"Recall@{k}"].append(float(hit))
            vals[f"NDCG@{k}"].append(1/np.log2(rank+1) if hit else 0.)
    return {k:float(np.mean(v)) for k,v in vals.items()}

def popularity_baseline(train_sequences,examples,num_items,ks=(5,10,20)):
    counts=np.zeros(num_items)
    for pairs in train_sequences.values():
        for item,_ in pairs: counts[item]+=1
    order=np.argsort(-counts); vals={f"Recall@{k}":[] for k in ks}|{f"NDCG@{k}":[] for k in ks}|{"MRR":[]}
    for ex in examples:
        seen=set(ex["history"]); target=int(ex["target"]); rank=1
        for item in order:
            if item in seen: continue
            if int(item)==target: break
            rank+=1
        vals["MRR"].append(1/rank)
        for k in ks:
            hit=rank<=k; vals[f"Recall@{k}"].append(float(hit)); vals[f"NDCG@{k}"].append(1/np.log2(rank+1) if hit else 0.)
    return {k:float(np.mean(v)) for k,v in vals.items()}
