
import pandas as pd,numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from trustlens.features.embeddings import encode
from trustlens.features.linguistic import frame

df=pd.read_csv("data/raw/trustlens.csv"); tr,te=train_test_split(df,test_size=.2,random_state=45,stratify=df.agreement)
s1,s2=encode(tr.scenario),encode(te.scenario); r1,r2=encode(tr.response),encode(te.response); l1,l2=frame(tr),frame(te)
sc=StandardScaler(); l1,l2=sc.fit_transform(l1),sc.transform(l2)
V={"response":(r1,r2),"scenario_response":(np.hstack([s1,r1]),np.hstack([s2,r2])),"scenario_response_linguistic":(np.hstack([s1,r1,l1]),np.hstack([s2,r2,l2])),"all_plus_perceived":(np.hstack([s1,r1,l1,tr[["perceived_ai"]]]),np.hstack([s2,r2,l2,te[["perceived_ai"]]]))}
rows=[]
for n,(a,b) in V.items():
 m=LogisticRegression(max_iter=1000,class_weight="balanced"); m.fit(a,tr.agreement); rows.append({"variant":n,"macro_f1":f1_score(te.agreement,m.predict(b),average="macro")})
Path("results/metrics").mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv("results/metrics/ablation.csv",index=False); print(pd.DataFrame(rows))
