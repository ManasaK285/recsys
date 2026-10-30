import pandas as pd
from src.generate_data import generate
from src.analytics import compare_releases,ask_analytics
def test_seeded_generator():pd.testing.assert_frame_equal(generate(4,8,7),generate(4,8,7))
def test_regression_is_injected():
 d=generate(5,14,42);assert d.is_injected_regression.any();assert d.loc[d.is_injected_regression,'launch_time_ms'].mean()>d.loc[~d.is_injected_regression,'launch_time_ms'].mean()
def test_release_comparison(tmp_path):
 p=tmp_path/'x.parquet';generate(5,14,42).to_parquet(p,index=False);r=compare_releases('build-18.1','build-18.2',p);x=next(i for i in r if i['metric']=='launch_time_ms');assert x['candidate_mean']>x['baseline_mean']
def test_analyst_intent(tmp_path):
 p=tmp_path/'x.parquet';generate(5,14,42).to_parquet(p,index=False);assert ask_analytics('Compare battery drain',p)['evidence']
