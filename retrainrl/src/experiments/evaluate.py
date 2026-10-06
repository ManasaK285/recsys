import json,pandas as pd
from stable_baselines3 import PPO
from src.utils.config import ROOT,load_config
from src.data.stream import TemporalStream
from src.environment.retraining_env import RetrainingEnv
from src.baselines.schedules import periodic,random_budget,drift_trigger
def run(e,a,name):
 o,_=e.reset();rows=[]
 for t,x in enumerate(a):
  o,r,d,_,i=e.step(x);rows.append({'strategy':name,'period':t+1,**i,'reward':r})
  if d:break
 return rows
def main():
 c=load_config();p=ROOT/c['data']['processed_dir'];r=json.load(open(p/'sequences.json'));mp=json.load(open(p/'mappings.json'));s=TemporalStream(r,mp['num_items'],c,c['seed']);base=RetrainingEnv(s,mp['num_users'],mp['num_items'],c);n=len(s.periods);b=c['environment']['retraining_budget'];ss={'never':[0]*n,'periodic_finetune':periodic(n,b,1),'periodic_sample':periodic(n,max(1,b//2),2),'random':random_budget(n,b,c['seed']),'drift_trigger':drift_trigger(base)}
 path=ROOT/'artifacts/checkpoints/ppo.zip'
 if path.exists():
  m=PPO.load(path);o,_=base.reset();a=[]
  for _ in range(n):
   x,_=m.predict(o,deterministic=True);o,_,d,_,_=base.step(int(x));a.append(int(x))
   if d:break
  ss['ppo_rtagent']=a
 rows=[]
 for name,a in ss.items():rows+=run(RetrainingEnv(s,mp['num_users'],mp['num_items'],c),a,name)
 out=ROOT/'artifacts/evaluation';out.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(rows);df.to_csv(out/'period_results.csv',index=False);summary=df.groupby('strategy').agg(mean_recall=('recall@10','mean'),mean_ndcg=('ndcg@10','mean'),mean_coverage=('coverage@10','mean'),mean_diversity=('diversity@10','mean'),total_cost=('cost','sum'),total_reward=('reward','sum')).reset_index();summary.to_csv(out/'results.csv',index=False);print(summary.to_string(index=False))
if __name__=='__main__':main()
