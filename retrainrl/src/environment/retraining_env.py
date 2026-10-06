import numpy as np,pandas as pd,gymnasium as gym
from gymnasium import spaces
from src.models.factory import build_model
from src.evaluation.metrics import evaluate
from src.monitoring.drift import compute_drift
from src.retraining.strategies import RetrainingManager,COSTS
class RetrainingEnv(gym.Env):
    def __init__(self,stream,nu,ni,cfg,seed=42):
        super().__init__()
        self.stream=stream
        self.nu=nu
        self.ni=ni
        self.cfg=cfg
        self.seedv=seed
        self.rng=np.random.default_rng(seed)

        self.action_space=spaces.Discrete(4)
        self.observation_space=spaces.Box(-10,10,(12,),np.float32)

        # Initialize state used by evaluation/baselines
        self.reference=self.stream.initial.copy()
    def reset(self,seed=None,options=None):
        super().reset(seed=seed);self.t=0;self.budget=self.cfg['environment']['retraining_budget'];self.age=0;self.last=0;self.model=build_model(self.cfg,self.nu,self.ni,self.seedv);self.reference=self.stream.initial.copy();self.history=self.reference.copy();self.model.fit(self.history);self.prev=evaluate(self.model,self.stream.periods[0]);return self._obs(self.stream.periods[0]),{}
    def _obs(self,c):
        m=evaluate(self.model,c);d=compute_drift(self.reference,c,self.ni);return np.array([m['recall@10'],m['ndcg@10'],m['coverage@10'],m['diversity@10'],m['ndcg@10']-self.prev['ndcg@10'],d,len(c)/max(1,len(self.reference)),c.item_id.nunique()/max(1,self.ni),self.age/12,self.age/12,self.budget/max(1,self.cfg['environment']['retraining_budget']),self.last/3],np.float32)
    def step(self,a):
        a=int(a);a=a if COSTS[a]<=self.budget else 0;c=self.stream.periods[self.t];cost=RetrainingManager(self.model,self.cfg,self.rng).apply(a,self.history);self.history=pd.concat([self.history,c],ignore_index=True);self.budget-=cost;self.age=0 if a else self.age+1;m=evaluate(self.model,c);d=compute_drift(self.reference,c,self.ni);q=.5*m['recall@10']+.5*m['ndcg@10'];dv=.5*m['coverage@10']+.5*m['diversity@10'];r=q+.15*dv-.12*(cost/4)-.08*(self.age/12)+.03*d*(1 if a else -1);self.prev=m;self.last=a;self.t+=1;done=self.t>=len(self.stream.periods);o=np.zeros(12,np.float32) if done else self._obs(self.stream.periods[self.t]);return o,float(r),done,False,{'action':a,'cost':cost,'budget':self.budget,'drift':d,**m}
