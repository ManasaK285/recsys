import numpy as np, torch
from src.model import SASRec,ASMoE
def test_forward():
    n,c,l=30,6,8; m=np.zeros(n,dtype=np.int64); m[2:]=np.random.randint(1,c,n-2)
    x=torch.randint(2,n,(4,l)); cats=torch.tensor(m[x.numpy()]); v=torch.ones_like(x,dtype=torch.bool)
    s,_=SASRec(n,c,m,l,16,4,1)(x,cats,v); assert s.shape==(4,n)
    model=ASMoE(n,c,m,l,16,4,1,4,2); s,z=model(x,cats,v); assert s.shape==(4,n) and z.shape==(4,16)
