import numpy as np
def dist(df,n):
    x=np.bincount(df.item_id.astype(int),minlength=n+1)[1:]+1e-12;return x/x.sum()
def compute_drift(a,b,n):
    p,q=dist(a,n),dist(b,n);m=(p+q)/2;return float(.5*np.sum(p*np.log(p/m))+.5*np.sum(q*np.log(q/m)))
