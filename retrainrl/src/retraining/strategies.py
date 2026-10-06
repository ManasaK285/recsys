import pandas as pd
COSTS=[0,1,2,4]
class RetrainingManager:
    def __init__(self,model,cfg,rng):self.model=model;self.cfg=cfg;self.rng=rng
    def apply(self,a,h):
        a=int(a)
        if a==0:return 0
        if a==1:
            n=max(1,int(len(h)*self.cfg['environment']['recent_fraction']));self.model.partial_fit(h.sort_values('timestamp').tail(n));return 1
        if a==2:
            n=max(1,int(len(h)*self.cfg['environment']['sample_fraction']));s=h.sample(min(n,len(h)),random_state=int(self.rng.integers(1_000_000)));r=h.sort_values('timestamp').tail(max(1,int(len(h)*.1)));self.model.partial_fit(pd.concat([s,r]).drop_duplicates());return 2
        self.model.fit(h);return 4
