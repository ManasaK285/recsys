import numpy as np
C=[0,1,2,4]
def periodic(n,b,a):
 o=[0]*n
 for t in range(0,n,max(1,n//max(1,b))):
  if sum(C[x] for x in o)+C[a]<=b:o[t]=a
 return o
def random_budget(n,b,seed=42):
 rng=np.random.default_rng(seed);o=[0]*n
 for t in rng.permutation(n):
  for a in rng.permutation([1,2,3]):
   if sum(C[x] for x in o)+C[a]<=b:o[t]=int(a);break
 return o
def drift_trigger(env,threshold=.08):
 from src.monitoring.drift import compute_drift
 o=[];spent=0
 for c in env.stream.periods:
  a=1 if compute_drift(env.reference,c,env.ni)>threshold and spent+1<=env.cfg['environment']['retraining_budget'] else 0;o.append(a);spent+=C[a]
 return o
