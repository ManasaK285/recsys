from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
from src.analytics import load_data,detect_anomalies
def main():
 d=load_data(); truth=d[['timestamp','device_id','is_injected_regression']].copy(); truth['timestamp']=truth.timestamp.dt.floor('h'); pred=detect_anomalies(limit=200000)
 if len(pred):
  pred=pred[['timestamp','device_id']].copy(); pred['timestamp']=pred.timestamp.dt.floor('h'); pred['predicted']=True; pred=pred.drop_duplicates(['timestamp','device_id']); truth=truth.merge(pred,on=['timestamp','device_id'],how='left'); truth.predicted=truth.predicted.fillna(False).astype(bool)
 else:truth['predicted']=False
 y,p=truth.is_injected_regression.astype(bool),truth.predicted.astype(bool); tn,fp,fn,tp=confusion_matrix(y,p,labels=[False,True]).ravel(); fpr=fp/(fp+tn) if fp+tn else 0
 print(f'Rows={len(truth):,} TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}'); print(f'Precision={precision_score(y,p,zero_division=0):.4f} Recall={recall_score(y,p,zero_division=0):.4f} F1={f1_score(y,p,zero_division=0):.4f} FPR={fpr:.4f}'); print('Synthetic benchmark only. Record machine and configuration with results.')
if __name__=='__main__':main()
