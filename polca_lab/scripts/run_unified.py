from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from polca_lab.core.evaluator import StochasticEvaluator
from polca_lab.core.optimizer import POLCAOptimizer
from polca_lab.core.metrics import auc,evals_to_threshold,mean_std
from polca_lab.providers.unified_oracle import UnifiedLocalOracle
from polca_lab.providers.openai_oracle import OpenAICompatibleOracle
from polca_lab.benchmarks.support_benchmark import SupportBenchmark
from polca_lab.benchmarks.rag_benchmark import RAGBenchmark
from polca_lab.benchmarks.agent_benchmark import AgentBenchmark
SEEDS={'prompt':['Answer customer requests directly and helpfully.'],'rag':['Answer using the provided context.'],'agent':['Handle requests safely and use the appropriate tools.']}
def make_benchmark(phase): return {'prompt':SupportBenchmark,'rag':RAGBenchmark,'agent':AgentBenchmark}[phase]()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['prompt','rag','agent','all'],default='all'); p.add_argument('--iterations',type=int,default=25); p.add_argument('--seeds',type=int,default=3); p.add_argument('--oracle',choices=['local','llm'],default='local'); p.add_argument('--noise',type=float,default=.10); p.add_argument('--out',default='results/unified_results.json'); a=p.parse_args()
 phases=['prompt','rag','agent'] if a.phase=='all' else [a.phase]; results=[]
 for phase in phases:
  for seed in range(1,a.seeds+1):
   bench=make_benchmark(phase); oracle=UnifiedLocalOracle(phase,seed) if a.oracle=='local' else OpenAICompatibleOracle(); cfg={'seed':seed,'iterations':a.iterations,'memory_size':24,'epsilon':.28,'use_epsilon_net':True,'use_summary':True,'batch_size':12,'proposals_per_iteration':3}
   opt=POLCAOptimizer(StochasticEvaluator(bench,a.noise,seed),oracle,cfg); opt.initialize(SEEDS[phase]); log=opt.run(); traj=[x['best_test_score'] for x in log]; best=max(traj,default=0); results.append({'phase':phase,'seed':seed,'evaluations':opt.evaluator.total_evals,'best_test':best,'auc':auc(traj),'evals_to_80':evals_to_threshold(traj,.80),'evals_to_90':evals_to_threshold(traj,.90),'final_memory':len(opt.memory.items),'trajectory':traj})
 summary={}
 for phase in phases:
  rows=[r for r in results if r['phase']==phase]; vals=[r['best_test'] for r in rows]; m,s=mean_std(vals); summary[phase]={'mean_best_test':m,'std_best_test':s,'mean_auc':sum(r['auc'] for r in rows)/len(rows),'mean_evaluations':sum(r['evaluations'] for r in rows)/len(rows)}
 out={'version':'unified-v3','results':results,'summary':summary}; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
