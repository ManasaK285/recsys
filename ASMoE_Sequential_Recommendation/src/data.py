import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
PAD_ITEM, MASK_ITEM = 0, 1

def generate_synthetic(num_users=800,num_items=500,num_categories=20,
                       min_interactions=18,max_interactions=32,seed=42):
    rng=np.random.default_rng(seed)
    item_categories=rng.integers(0,num_categories,size=num_items)
    item_pop=rng.lognormal(0,0.8,size=num_items); item_pop/=item_pop.sum()
    rows=[]; ts=1
    for u in range(num_users):
        n=int(rng.integers(min_interactions,max_interactions+1))
        alpha=np.ones(num_categories)*.35
        pref=rng.choice(num_categories,size=min(3,num_categories),replace=False)
        alpha[pref]+=4
        user_cat=rng.dirichlet(alpha)
        cat=int(rng.choice(num_categories,p=user_cat))
        for t in range(n):
            if t:
                p=.05*user_cat
                p[cat]+=.62; p[(cat-1)%num_categories]+=.16; p[(cat+1)%num_categories]+=.16
                p/=p.sum(); cat=int(rng.choice(num_categories,p=p))
            cand=np.where(item_categories==cat)[0]
            w=item_pop[cand]; w/=w.sum()
            item=int(rng.choice(cand,p=w))
            rows.append((u,item,ts,cat)); ts+=1
    return pd.DataFrame(rows,columns=["user_id","item_id","timestamp","category"])

def load_csv(path):
    df=pd.read_csv(path)
    req={"user_id","item_id","timestamp"}
    if not req.issubset(df.columns): raise ValueError(f"Missing columns: {sorted(req-set(df.columns))}")
    if "category" not in df: df["category"]=df["item_id"].map(lambda x: hash(str(x))%20)
    return df[["user_id","item_id","timestamp","category"]]

def encode_and_split(df,min_item_frequency=1):
    df=df.sort_values(["user_id","timestamp"]).reset_index(drop=True)
    keep=set(df.item_id.value_counts().loc[lambda x:x>=min_item_frequency].index)
    df=df[df.item_id.isin(keep)].copy()
    iv=sorted(df.item_id.unique()); item_to_idx={v:i+2 for i,v in enumerate(iv)}
    df["item_idx"]=df.item_id.map(item_to_idx)
    cv=sorted(df.category.unique()); cat_to_idx={v:i+1 for i,v in enumerate(cv)}
    df["category_idx"]=df.category.map(cat_to_idx)
    seq={}
    for uid,g in df.groupby("user_id",sort=False):
        pairs=list(zip(g.item_idx.tolist(),g.category_idx.tolist()))
        if len(pairs)>=4: seq[uid]=pairs
    train={}; val={}; test={}
    for uid,pairs in seq.items():
        train[uid]=pairs[:-2]; val[uid]=pairs[-2]; test[uid]=pairs[-1]
    item_to_category=np.zeros(len(item_to_idx)+2,dtype=np.int64)
    for raw,idx in item_to_idx.items():
        item_to_category[idx]=int(df.loc[df.item_id==raw,"category_idx"].iloc[0])
    return dict(train=train,val=val,test=test,item_to_idx=item_to_idx,
                cat_to_idx=cat_to_idx,item_to_category=item_to_category,
                num_items=len(item_to_idx)+2,num_categories=len(cat_to_idx)+1)

class SequenceDataset(Dataset):
    def __init__(self,train_sequences,max_len):
        self.examples=[]
        for pairs in train_sequences.values():
            items=[p[0] for p in pairs]; cats=[p[1] for p in pairs]
            for end in range(1,len(items)):
                start=max(0,end-max_len)
                self.examples.append((items[start:end],cats[start:end],items[end]))
    def __len__(self): return len(self.examples)
    def __getitem__(self,i): return self.examples[i]

def collate_batch(batch,max_len):
    b=len(batch); x=torch.zeros((b,max_len),dtype=torch.long)
    c=torch.zeros_like(x); valid=torch.zeros_like(x,dtype=torch.bool); y=torch.zeros(b,dtype=torch.long)
    for i,(items,cats,target) in enumerate(batch):
        n=min(len(items),max_len); x[i,-n:]=torch.tensor(items[-n:]); c[i,-n:]=torch.tensor(cats[-n:])
        valid[i,-n:]=True; y[i]=target
    return x,c,valid,y

def make_eval_examples(split,history_source,max_len):
    out=[]
    for uid,target_pair in split.items():
        hp=history_source[uid]; items=[p[0] for p in hp]; cats=[p[1] for p in hp]
        out.append({"uid":uid,"history":items[-max_len:],"history_categories":cats[-max_len:],"target":target_pair[0]})
    return out
