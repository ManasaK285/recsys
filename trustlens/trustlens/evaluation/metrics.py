
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
def report(y,p,prob,name):
    return {"model":name,"accuracy":accuracy_score(y,p),"macro_f1":f1_score(y,p,average="macro"),"precision":precision_score(y,p,zero_division=0),"recall":recall_score(y,p,zero_division=0),"roc_auc":roc_auc_score(y,prob)}
