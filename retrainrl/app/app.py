from pathlib import Path
import pandas as pd,streamlit as st
ROOT=Path(__file__).resolve().parents[1];st.set_page_config(page_title='ReTrainRL',layout='wide');st.title('ReTrainRL');st.caption('Adaptive retraining under distribution drift and a global compute budget')
f=ROOT/'artifacts/evaluation/results.csv';p=ROOT/'artifacts/evaluation/period_results.csv'
if not f.exists():st.warning('Run python -m src.experiments.evaluate first.');st.stop()
df=pd.read_csv(f);st.dataframe(df,use_container_width=True)
if p.exists():
 d=pd.read_csv(p);name=st.selectbox('Strategy',sorted(d.strategy.unique()));x=d[d.strategy==name].set_index('period');st.line_chart(x[['recall@10','ndcg@10']]);st.line_chart(x[['drift']]);st.dataframe(x[['action','cost','budget']],use_container_width=True)
