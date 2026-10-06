import pandas as pd
from src.monitoring.drift import compute_drift
def test_drift():
 x=pd.DataFrame({'item_id':[1,2,2,3]});assert compute_drift(x,x,3)<1e-8
