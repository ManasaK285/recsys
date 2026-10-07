import numpy as np
import pandas as pd

def empirical_conditional_relationship(df,tasks,smoothing=1.0):
    y=df[tasks].to_numpy(dtype=np.float64)
    r=np.zeros((len(tasks),len(tasks)))
    for i in range(len(tasks)):
        mask=y[:,i]==1
        count=mask.sum()
        for j in range(len(tasks)):
            joint=((y[:,i]==1)&(y[:,j]==1)).sum()
            r[i,j]=(joint+smoothing)/(count+2*smoothing)
    return pd.DataFrame(r,index=tasks,columns=tasks)
