import pandas as pd
import streamlit as st
import plotly.express as px
from src.data.synthetic_raters import make_synthetic_data,split_by_prompt
from src.training.train import train_models
from src.evaluation.metrics import evaluate_probabilities

st.set_page_config(page_title="AdaptiveReward Lab",layout="wide")
st.title("AdaptiveReward Lab")
st.caption("Research-inspired experiments for personalized reward models.")
st.warning("Synthetic experiment results are not claims about real human preferences.")
with st.sidebar:
    users=st.slider("Number of users",10,100,30,10)
    n=st.slider("Preference comparisons",300,3000,900,300)
    h=st.slider("Preference heterogeneity",0.0,1.5,.8,.1)
    epochs=st.slider("Training epochs",30,300,100,10)
    seed=st.number_input("Random seed",0,9999,42)
    go=st.button("Train and evaluate",type="primary")
if go:
    with st.spinner("Training models..."):
        data,_=make_synthetic_data(n_users=users,n_prompts=n,heterogeneity=h,seed=int(seed))
        tr,te=split_by_prompt(data,seed=int(seed))
        models=train_models(tr,n_users=users,epochs=epochs,seed=int(seed))
        rows=[]
        for name,m in models.items():
            p=m.predict_proba(te.features_a,te.features_b,te.user_ids)
            rows.append({"model":name,**evaluate_probabilities(te.labels,p)})
        st.session_state["results"]=pd.DataFrame(rows)
if "results" in st.session_state:
    df=st.session_state["results"]
    st.dataframe(df,use_container_width=True)
    c1,c2=st.columns(2)
    c1.plotly_chart(px.bar(df,x="model",y="accuracy",title="Accuracy (higher is better)"),use_container_width=True)
    c2.plotly_chart(px.bar(df,x="model",y="log_loss",title="Log loss (lower is better)"),use_container_width=True)
    st.download_button("Download metrics CSV",df.to_csv(index=False),"model_comparison.csv","text/csv")
else:
    st.info("Choose experiment settings and click Train and evaluate.")
st.markdown("---")
st.markdown("**Method:** shared reward features plus user-specific weights. This is a research prototype, not a deployed preference or safety judge.")
