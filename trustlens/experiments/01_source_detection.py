
from pathlib import Path
import joblib,pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from trustlens.data.loader import load_data
from trustlens.features.embeddings import encode
from trustlens.features.linguistic import frame
from trustlens.models.tfidf import build
from trustlens.evaluation.metrics import report

df=load_data(); tr,te=train_test_split(df,test_size=.2,random_state=42,stratify=df.source)
ytr=(tr.source=="ai").astype(int); yte=(te.source=="ai").astype(int); rows=[]
m=build(); m.fit(tr.response,ytr); p=m.predict(te.response); rows.append(report(yte,p,m.predict_proba(te.response)[:,1],"TF-IDF + Logistic Regression"))
Xtr,Xte=encode(tr.response),encode(te.response); m=LogisticRegression(max_iter=1000,class_weight="balanced"); m.fit(Xtr,ytr); p=m.predict(Xte); rows.append(report(yte,p,m.predict_proba(Xte)[:,1],"Sentence-BERT + Logistic Regression"))
a,b=frame(tr),frame(te); sc=StandardScaler(); a,b=sc.fit_transform(a),sc.transform(b); m=LogisticRegression(max_iter=1000,class_weight="balanced"); m.fit(a,ytr); p=m.predict(b); rows.append(report(yte,p,m.predict_proba(b)[:,1],"Linguistic features + Logistic Regression"))
Path("results/metrics").mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv("results/metrics/source_detection.csv",index=False)
pd.DataFrame({"feature":frame(tr).columns,"importance":abs(m.coef_[0])}).to_csv("results/metrics/linguistic_importance.csv",index=False)
print(pd.DataFrame(rows).to_string(index=False))
