
from pathlib import Path
import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from trustlens.features.embeddings import encode
from trustlens.features.linguistic import frame
from trustlens.evaluation.metrics import report

df=pd.read_csv("data/raw/trustlens.csv"); tr,te=train_test_split(df,test_size=.2,random_state=43,stratify=df.agreement)
se1,se2=encode(tr.scenario),encode(te.scenario); re1,re2=encode(tr.response),encode(te.response)
l1,l2=frame(tr),frame(te); sc=StandardScaler(); l1,l2=sc.fit_transform(l1),sc.transform(l2)
rows=[]
for name,a,b in [
("Response embedding",re1,re2),
("Scenario + response",np.hstack([se1,re1]),np.hstack([se2,re2])),
("Scenario + response + linguistic",np.hstack([se1,re1,l1]),np.hstack([se2,re2,l2])),
("All + perceived source",np.hstack([se1,re1,l1,tr[["perceived_ai"]]]),np.hstack([se2,re2,l2,te[["perceived_ai"]]]))]:
    m=LogisticRegression(max_iter=1000,class_weight="balanced"); m.fit(a,tr.agreement); p=m.predict(b)
    rows.append(report(te.agreement,p,m.predict_proba(b)[:,1],name))
Path("results/metrics").mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv("results/metrics/agreement.csv",index=False); print(pd.DataFrame(rows).to_string(index=False))
