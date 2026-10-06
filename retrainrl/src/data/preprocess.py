import json,pandas as pd
from src.utils.config import ROOT,load_config
def main():
    c=load_config();raw=ROOT/c['data']['raw_dir']/'ml-1m'/'ratings.dat';out=ROOT/c['data']['processed_dir'];out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with open(raw,encoding='latin-1') as f:
        for line in f:
            u,i,r,t=line.strip().split('::');rows.append((int(u),int(i),float(r),int(t)))
    df=pd.DataFrame(rows,columns=['user_id','item_id','rating','timestamp']).sort_values('timestamp')
    good=df.user_id.value_counts();good=good[good>=c['data']['min_interactions']].index;df=df[df.user_id.isin(good)]
    df=df[df.user_id.isin(sorted(df.user_id.unique())[:c['data']['max_users']])]
    um = {u: i for i, u in enumerate(sorted(df.user_id.unique()))}
    im = {raw: i for i, raw in enumerate(sorted(df.item_id.unique()))}

    df["user_id"] = df.user_id.map(um)
    df["item_id"] = df.item_id.map(im)
    rec=[]
    for u,g in df.groupby('user_id'):
        g=g.sort_values('timestamp');rec.append({'user_id':int(u),'interactions':[{'item_id':int(i),'rating':float(r),'timestamp':int(t)} for i,r,t in zip(g.item_id,g.rating,g.timestamp)]})
    json.dump(rec,open(out/'sequences.json','w'));json.dump({'num_users':len(um),'num_items':len(im)},open(out/'mappings.json','w'));print(f'Users={len(um)} Items={len(im)}')
if __name__=='__main__':main()
