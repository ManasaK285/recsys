import requests,pandas as pd,plotly.express as px,streamlit as st
st.set_page_config(page_title='PowerPulse AI',page_icon='⚡',layout='wide'); API=st.sidebar.text_input('API URL','http://127.0.0.1:8000').rstrip('/'); st.sidebar.title('⚡ PowerPulse AI')
def get(p,params=None):r=requests.get(API+p,params=params,timeout=30);r.raise_for_status();return r.json()
def post(p,data):r=requests.post(API+p,json=data,timeout=60);r.raise_for_status();return r.json()
st.title('PowerPulse AI');st.caption('Device telemetry • Performance regression • Evidence-grounded AI')
try:s=get('/analytics/summary')
except Exception as e:st.error(f'API unavailable. Start uvicorn src.api:app --reload. {e}');st.stop()
c=st.columns(4);c[0].metric('Observations',f"{s['observations']:,}");c[1].metric('Devices',s['devices']);c[2].metric('Models',s['models']);c[3].metric('Releases',len(s['releases']));st.caption(f"{s['time_start']} — {s['time_end']}")
t1,t2,t3,t4=st.tabs(['Overview','Anomalies','Release comparison','AI analyst'])
with t1:
 st.write(f"Mean CPU: **{s['mean_cpu_pct']}%** · Mean battery drain: **{s['mean_battery_drain_pct_hr']}%/hour**"); a=get('/analytics/anomalies',{'limit':300})['items']
 if a:
  d=pd.DataFrame(a); m=st.selectbox('Metric',sorted(d.metric.unique())); st.plotly_chart(px.scatter(d[d.metric==m],x='timestamp',y='observed',color='severity',hover_data=['device_id','device_model','release','baseline','robust_z']),use_container_width=True)
with t2:
 d=pd.DataFrame(get('/analytics/anomalies',{'limit':1000})['items'])
 if len(d):
  sev=st.multiselect('Severity',sorted(d.severity.unique()),default=sorted(d.severity.unique())); st.dataframe(d[d.severity.isin(sev)],use_container_width=True)
with t3:
 r=get('/analytics/releases')['releases']
 if len(r)>1:
  a,b=st.columns(2); base=a.selectbox('Baseline',r,index=0); cand=b.selectbox('Candidate',r,index=len(r)-1)
  if base!=cand:
   d=pd.DataFrame(get('/analytics/releases/compare',{'baseline':base,'candidate':cand})['metrics']);st.dataframe(d,use_container_width=True);st.plotly_chart(px.bar(d,x='metric',y='percent_change',color='performance_interpretation'),use_container_width=True)
with t4:
 q=st.text_input('Ask the performance analyst','Which devices regressed after the latest release?')
 if st.button('Investigate',type='primary'):
  z=post('/analytics/ask',{'question':q});st.write(z['answer']);st.info(z['caveat']);
  if z.get('evidence'):st.dataframe(pd.DataFrame(z['evidence']),use_container_width=True)
  if z.get('simulator_affected_devices'):st.caption('Simulator ground truth for evaluation: '+', '.join(z['simulator_affected_devices']))
