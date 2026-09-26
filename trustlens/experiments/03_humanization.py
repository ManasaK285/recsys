
import random,pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from trustlens.features.embeddings import encode

def humanize(x,i):
    x=x.replace("The most defensible answer is that ","").replace("A reasonable conclusion is that ","").replace("Therefore, ","")
    if i%2==0: x=x.replace("important","imp")
    return " ".join(x.split())
df=pd.read_csv("data/raw/trustlens.csv"); ai=df[df.source=="ai"]; hu=df[df.source=="human"].sample(len(ai),random_state=44)
tr1,te1=train_test_split(ai,test_size=.3,random_state=44); tr=pd.concat([tr1,hu]); y=(tr.source=="ai").astype(int)
m=LogisticRegression(max_iter=1000,class_weight="balanced"); m.fit(encode(tr.response),y)
rows=[]
for name,texts in [("original_ai",te1.response.tolist()),("humanized_ai",[humanize(x,i) for i,x in enumerate(te1.response)])]:
    p=m.predict(encode(texts)); rows.append({"variant":name,"ai_recall":p.mean(),"n":len(p)})
Path("results/metrics").mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv("results/metrics/humanization.csv",index=False); print(pd.DataFrame(rows))
