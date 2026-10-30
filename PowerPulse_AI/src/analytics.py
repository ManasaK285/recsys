import numpy as np, pandas as pd, duckdb
from src.config import DATA_PATH
from src.generate_data import generate
METRICS=['cpu_pct','gpu_pct','memory_mb','battery_drain_pct_hr','launch_time_ms','fps','thermal_c','network_latency_ms']
HIGHER_BAD=set(METRICS)-{'fps'}
def load_data(path=DATA_PATH):
    if not path.exists(): path.parent.mkdir(parents=True,exist_ok=True); generate().to_parquet(path,index=False)
    d=pd.read_parquet(path); d['timestamp']=pd.to_datetime(d.timestamp,utc=True); return d
def summary(path=DATA_PATH):
    d=load_data(path); return {'observations':len(d),'devices':d.device_id.nunique(),'models':d.device_model.nunique(),'regions':d.region.nunique(),'releases':sorted(d.release.unique().tolist()),'time_start':d.timestamp.min().isoformat(),'time_end':d.timestamp.max().isoformat(),'mean_cpu_pct':round(float(d.cpu_pct.mean()),2),'mean_battery_drain_pct_hr':round(float(d.battery_drain_pct_hr.mean()),3)}
def detect_anomalies(path=DATA_PATH,limit=300):
    d=load_data(path).sort_values('timestamp'); out=[]
    for m in METRICS:
        g=d.groupby('device_id',sort=False)[m]; base=g.transform(lambda s:s.shift(1).rolling(24,min_periods=8).median())
        mad=g.transform(lambda s:s.shift(1).rolling(24,min_periods=8).apply(lambda x:np.median(np.abs(x-np.median(x))),raw=True))
        z=((d[m]-base)/(1.4826*mad.replace(0,np.nan))).replace([np.inf,-np.inf],np.nan).fillna(0); mask=z.abs()>=3.5
        if mask.any():
            x=d.loc[mask,['timestamp','device_id','device_model','region','release','is_injected_regression']].copy(); x['metric']=m; x['observed']=d.loc[mask,m].astype(float).values; x['baseline']=base.loc[mask].astype(float).values; x['robust_z']=z.loc[mask].values; x['severity']=np.where(x.robust_z.abs()>=6,'critical',np.where(x.robust_z.abs()>=4.5,'high','medium')); out.append(x)
    if not out:return pd.DataFrame(columns=['timestamp','device_id','device_model','region','release','metric','observed','baseline','robust_z','severity','is_injected_regression'])
    r=pd.concat(out,ignore_index=True).sort_values('robust_z',key=lambda s:s.abs(),ascending=False); return r.head(limit).reset_index(drop=True)
def compare_releases(baseline,candidate,path=DATA_PATH):
    d=load_data(path); available=set(d.release.unique())
    if baseline not in available or candidate not in available: raise ValueError(f'Unknown release. Available: {sorted(available)}')
    a,b=d[d.release==baseline],d[d.release==candidate]; rows=[]
    for m in METRICS:
        x,y=float(a[m].mean()),float(b[m].mean()); delta=y-x; pct=delta/abs(x)*100 if x else None
        worse=(m in HIGHER_BAD and delta>0) or (m=='fps' and delta<0)
        rows.append({'metric':m,'baseline_mean':round(x,4),'candidate_mean':round(y,4),'absolute_change':round(delta,4),'percent_change':round(pct,2) if pct is not None else None,'direction':'increase' if delta>0 else 'decrease' if delta<0 else 'unchanged','performance_interpretation':'worse' if worse else 'better' if delta else 'unchanged'})
    return rows
def investigate(device_id,path=DATA_PATH):
    d=load_data(path); x=d[d.device_id==device_id].sort_values('timestamp')
    if x.empty: raise ValueError(f'Unknown device: {device_id}')
    latest=x.release.iloc[-1]; before=x[x.release!=latest]; after=x[x.release==latest]; ev=[]
    for m in METRICS:
        a,b=float(before[m].mean()),float(after[m].mean()); ev.append({'metric':m,'previous_release_mean':round(a,3),'latest_release_mean':round(b,3),'percent_change':round((b-a)/abs(a)*100,2) if a else None})
    an=detect_anomalies(path,10000); count=int((an.device_id==device_id).sum()) if len(an) else 0
    return {'device_id':device_id,'device_model':str(x.device_model.iloc[0]),'latest_release':str(latest),'observations':len(x),'anomaly_records_in_top_10000':count,'evidence':ev,'hypotheses':[{'title':'Release-associated workload regression','evidence':'Several measurements changed after the build transition.','caveat':'Temporal association is not proof of causation.'},{'title':'Resource pressure may contribute','evidence':'Inspect CPU, GPU, memory and thermal measurements together.','caveat':'Validate with traces, logs and controlled experiments.'}],'next_checks':['Compare startup traces and CPU profiles across builds.','Inspect memory allocation and memory-pressure events.','Compare battery drain under matched workloads.','Check thermal state and throttling before interpreting FPS changes.','Compare affected and unaffected devices of the same model.']}
def ask_analytics(question,path=DATA_PATH):
    q=question.lower(); d=load_data(path)
    if any(t in q for t in ['regress','latest release','build comparison']):
        r=sorted(d.release.unique()); ev=compare_releases(r[-2],r[-1],path); affected=sorted(d.loc[d.is_injected_regression,'device_id'].unique().tolist())
        return {'question':question,'answer':f'Compared {r[-1]} with {r[-2]}. Metrics are computed from release cohorts. Simulator ground truth labels {len(affected)} affected devices for evaluation.','evidence':ev,'simulator_affected_devices':affected,'caveat':'A release-associated change is not causal proof.'}
    if 'anomal' in q or 'outlier' in q:
        a=detect_anomalies(path,30).drop(columns=['is_injected_regression'],errors='ignore'); return {'question':question,'answer':f'Returned {len(a)} highest-ranked anomaly records.','evidence':a.astype(str).to_dict('records'),'caveat':'Statistical anomalies require engineering validation.'}
    if 'battery' in q:
        a=d.groupby(['device_model','release']).battery_drain_pct_hr.mean().reset_index(); return {'question':question,'answer':'Average battery drain by device model and release.','evidence':a.round(3).to_dict('records'),'caveat':'Synthetic cohort averages; workload matching is required for real comparisons.'}
    if any(t in q for t in ['device','launch','fps','cpu']):
        m='launch_time_ms' if 'launch' in q else 'fps' if 'fps' in q else 'cpu_pct'; a=d.groupby(['device_id','device_model'])[m].mean().reset_index().sort_values(m,ascending=m!='fps').head(10); return {'question':question,'answer':f'Top 10 devices by mean {m}.','evidence':a.round(3).to_dict('records'),'caveat':'Descriptive aggregate over synthetic observations.'}
    return {'question':question,'answer':'I can analyze release regressions, anomalies, battery drain, launch time, FPS and CPU metrics.','evidence':[],'caveat':"Try: 'Which devices regressed after the latest release?', 'Show anomalies', or 'Compare battery drain by device model'."}
def duckdb_model_summary(path=DATA_PATH):
    load_data(path); con=duckdb.connect(':memory:')
    try:
        p=str(path).replace("'","''"); return con.execute(f"SELECT device_model, COUNT(*) observations, AVG(cpu_pct) mean_cpu, AVG(battery_drain_pct_hr) mean_battery FROM read_parquet('{p}') GROUP BY device_model ORDER BY mean_cpu DESC").df().round(3).to_dict('records')
    finally: con.close()
