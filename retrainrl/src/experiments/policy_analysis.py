import json,pandas as pd,matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.utils.config import ROOT,load_config
from src.data.stream import TemporalStream
from src.environment.retraining_env import RetrainingEnv
def main():
 c=load_config();p=ROOT/c['data']['processed_dir'];r=json.load(open(p/'sequences.json'));mp=json.load(open(p/'mappings.json'));s=TemporalStream(r,mp['num_items'],c,c['seed']);e=RetrainingEnv(s,mp['num_users'],mp['num_items'],c);m=PPO.load(ROOT/'artifacts/checkpoints/ppo.zip');o,_=e.reset();rows=[]
 for t in range(len(s.periods)):
  a,_=m.predict(o,deterministic=True);o,reward,d,_,i=e.step(int(a));rows.append({'period':t+1,'action':int(a),'drift':i['drift'],'reward':reward})
  if d:break
 df=pd.DataFrame(rows);print(df.to_string(index=False));out=ROOT/'artifacts/figures';out.mkdir(parents=True,exist_ok=True);plt.figure(figsize=(9,4));plt.plot(df.period,df.drift,marker='o');plt.xlabel('Period');plt.ylabel('JS drift');plt.tight_layout();plt.savefig(out/'drift_curve.png',dpi=150);plt.close();plt.figure(figsize=(9,4));plt.step(df.period,df.action,where='mid');plt.yticks([0,1,2,3],['SKIP','FINE-TUNE','SAMPLE','FULL']);plt.xlabel('Period');plt.ylabel('Action');plt.tight_layout();plt.savefig(out/'policy_actions.png',dpi=150);plt.close()
if __name__=='__main__':main()
