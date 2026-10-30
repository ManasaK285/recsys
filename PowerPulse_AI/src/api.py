import logging,pandas as pd
from datetime import datetime,timezone
from fastapi import FastAPI,HTTPException,Query
from prometheus_client import Counter,Histogram,make_asgi_app
from src.analytics import summary,detect_anomalies,compare_releases,investigate,ask_analytics,load_data
from src.config import DATA_PATH
from src.schemas import TelemetryEvent,AskRequest
logging.basicConfig(level=logging.INFO); log=logging.getLogger('powerpulse')
REQ=Counter('powerpulse_requests_total','API requests',['path']); LAT=Histogram('powerpulse_request_seconds','Request duration',['path'])
app=FastAPI(title='PowerPulse AI',version='1.0.0',description='Device power and performance telemetry analytics'); app.mount('/metrics',make_asgi_app())
@app.middleware('http')
async def metrics(request,call_next):
    with LAT.labels(path=request.url.path).time(): REQ.labels(path=request.url.path).inc(); return await call_next(request)
@app.get('/health')
def health(): return {'status':'ok','service':'powerpulse-ai','timestamp':datetime.now(timezone.utc).isoformat()}
@app.get('/analytics/summary')
def get_summary(): return summary()
@app.get('/analytics/anomalies')
def anomalies(limit:int=Query(100,ge=1,le=2000)):
    d=detect_anomalies(limit); d=d.drop(columns=['is_injected_regression'],errors='ignore'); d['timestamp']=d.timestamp.astype(str) if len(d) else d.get('timestamp'); return {'count':len(d),'items':d.to_dict('records')}
@app.get('/analytics/releases')
def releases(): return {'releases':sorted(load_data().release.unique().tolist())}
@app.get('/analytics/releases/compare')
def compare(baseline:str,candidate:str):
    try:return {'baseline':baseline,'candidate':candidate,'metrics':compare_releases(baseline,candidate)}
    except ValueError as e:raise HTTPException(400,str(e))
@app.get('/analytics/investigate/{device_id}')
def device(device_id:str):
    try:return investigate(device_id)
    except ValueError as e:raise HTTPException(404,str(e))
@app.post('/analytics/ask')
def ask(body:AskRequest):return ask_analytics(body.question)
@app.post('/telemetry/ingest')
def ingest(events:list[TelemetryEvent]):
    if not events:raise HTTPException(400,'At least one event is required')
    if len(events)>5000:raise HTTPException(413,'Maximum batch size is 5000')
    old=load_data(); new=pd.DataFrame([e.model_dump() for e in events]); new['timestamp']=pd.to_datetime(new.timestamp,utc=True); new['is_injected_regression']=False
    merged=pd.concat([old,new],ignore_index=True); DATA_PATH.parent.mkdir(parents=True,exist_ok=True); merged.to_parquet(DATA_PATH,index=False); log.info('Accepted %s telemetry records',len(new)); return {'accepted':len(new),'total_observations':len(merged)}
