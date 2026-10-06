import json
from src.utils.config import ROOT,load_config
from src.data.stream import TemporalStream
from src.environment.retraining_env import RetrainingEnv
from src.agents.train import train
def main():
 c=load_config();p=ROOT/c['data']['processed_dir']
 if not (p/'sequences.json').exists():
  from src.data.download import main as d;from src.data.preprocess import main as pp;d();pp()
 r=json.load(open(p/'sequences.json'));mp=json.load(open(p/'mappings.json'));s=TemporalStream(r,mp['num_items'],c,c['seed']);e=RetrainingEnv(s,mp['num_users'],mp['num_items'],c);train(e,c,ROOT/'artifacts/checkpoints/ppo')
if __name__=='__main__':main()
