import numpy as np
def recall(r,t,k=10):return float(t in r[:k])
def ndcg(r,t,k=10):
    try:rank=r[:k].index(t)+1
    except ValueError:return 0.0
    return float(1/np.log2(rank+1))
def coverage(rs,n):return len(set(x for r in rs for x in r))/max(1,n)
def diversity(rs):
    vals=[]
    for r in rs:
        if len(r)>1:vals.append(np.mean([abs(r[i]-r[j])/max(r[i],r[j]) for i in range(len(r)) for j in range(i+1,len(r))]))
    return float(np.mean(vals)) if vals else 0.0
def evaluate(model,df,k=10,max_users=250):
    if df.empty:return {'recall@10':0.,'ndcg@10':0.,'coverage@10':0.,'diversity@10':0.}
    targets=df.sort_values('timestamp').groupby('user_id').tail(1)
    if len(targets)>max_users:targets=targets.sample(max_users,random_state=42)
    rs=[];rc=[];nd=[]
    for x in targets.itertuples():
        seen=set(df[(df.user_id==x.user_id)&(df.timestamp<x.timestamp)].item_id);r=model.recommend(x.user_id,k,seen);rs.append(r);rc.append(recall(r,x.item_id,k));nd.append(ndcg(r,x.item_id,k))
    return {'recall@10':float(np.mean(rc)),'ndcg@10':float(np.mean(nd)),'coverage@10':coverage(rs,model.num_items),'diversity@10':diversity(rs)}
