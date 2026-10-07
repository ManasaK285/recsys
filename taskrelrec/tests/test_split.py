import pandas as pd
from src.data.split import split_by_user

def test_no_user_overlap():
    df=pd.DataFrame({"user_id":[0,0,1,1,2,2,3,3],"item_id":range(8),
        "liked":[0,1,0,1,0,1,0,1],"strong_preference":[0]*8,
        "repeat_interest":[0]*8,"high_engagement":[0]*8})
    a,b,c=split_by_user(df,1,.5,.25)
    assert set(a.user_id).isdisjoint(b.user_id)
    assert set(a.user_id).isdisjoint(c.user_id)
    assert set(b.user_id).isdisjoint(c.user_id)
