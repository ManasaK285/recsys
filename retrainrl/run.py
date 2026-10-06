import subprocess,sys
from pathlib import Path
def r(m):subprocess.check_call([sys.executable,'-m',m])
if __name__=='__main__':
 p=Path(__file__).parent/'data/processed'
 if not (p/'sequences.json').exists():r('src.data.download');r('src.data.preprocess')
 r('src.experiments.train_ppo');r('src.experiments.evaluate');r('src.experiments.policy_analysis')
