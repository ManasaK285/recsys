from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from polca_lab.core.evaluator import StochasticEvaluator
from polca_lab.core.optimizer import POLCAOptimizer
from polca_lab.providers.unified_oracle import UnifiedLocalOracle
from polca_lab.benchmarks.support_benchmark import SupportBenchmark
from polca_lab.benchmarks.rag_benchmark import RAGBenchmark
from polca_lab.benchmarks.agent_benchmark import AgentBenchmark
B={'prompt':SupportBenchmark,'rag':RAGBenchmark,'agent':AgentBenchmark}; S={'prompt':['Answer customer requests directly and helpfully.'],'rag':['Answer using the provided context.'],'agent':['Handle requests safely and use the appropriate tools.']}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--phase',choices=B,default='prompt'); p.add_argument('--iterations',type=int,default=20); p.add_argument('--seeds',type=int,default=3); p.add_argument('--out',default='results/noise_sweep.json'); a=p.parse_args(); rows=[]
 for noise in (0,.05,.10,.20,.30,.40):
  for seed in range(1,a.seeds+1):
   b=B[a.phase](); o=POLCAOptimizer(StochasticEvaluator(b,noise,seed),UnifiedLocalOracle(a.phase,seed),{'seed':seed,'iterations':a.iterations,'memory_size':24,'epsilon':.28,'use_epsilon_net':True,'use_summary':True,'batch_size':12,'proposals_per_iteration':3}); o.initialize(S[a.phase]); log=o.run(); rows.append({'phase':a.phase,'noise':noise,'seed':seed,'evaluations':o.evaluator.total_evals,'best_test':max(x['best_test_score'] for x in log),'best_train':max(x['best_train_mean'] for x in log)})
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
