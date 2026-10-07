from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from polca_lab.core.evaluator import StochasticEvaluator
from polca_lab.core.optimizer import POLCAOptimizer
from polca_lab.core.metrics import auc,evals_to_threshold,mean_std
from polca_lab.providers.unified_oracle import UnifiedLocalOracle
from polca_lab.benchmarks.support_benchmark import SupportBenchmark
from polca_lab.benchmarks.rag_benchmark import RAGBenchmark
from polca_lab.benchmarks.agent_benchmark import AgentBenchmark
BENCH={'prompt':SupportBenchmark,'rag':RAGBenchmark,'agent':AgentBenchmark}
SEED={'prompt':['Answer customer requests directly and helpfully.'],'rag':['Answer using the provided context.'],'agent':['Handle requests safely and use the appropriate tools.']}
def run(phase,seed,iterations,noise,variant):
 b=BENCH[phase](); cfg={'seed':seed,'iterations':iterations,'memory_size':24,'epsilon':.28,'use_epsilon_net':variant!='no_epsilon','use_summary':variant!='no_summary','priority':'ucb' if variant=='ucb' else 'mean','batch_size':12,'proposals_per_iteration':3}; o=POLCAOptimizer(StochasticEvaluator(b,noise,seed),UnifiedLocalOracle(phase,seed),cfg); o.initialize(SEED[phase]); log=o.run(); tr=[x['best_test_score'] for x in log]; return {'phase':phase,'seed':seed,'variant':variant,'evaluations':o.evaluator.total_evals,'best_test':max(tr,default=0),'auc':auc(tr),'evals_to_80':evals_to_threshold(tr,.80),'evals_to_90':evals_to_threshold(tr,.90),'memory_size':len(o.memory.items),'trajectory':tr}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--iterations',type=int,default=20); p.add_argument('--seeds',type=int,default=3); p.add_argument('--noise',type=float,default=.10); p.add_argument('--out',default='results/research_suite.json'); a=p.parse_args(); rows=[]
 for phase in BENCH:
  for variant in ('full','no_epsilon','no_summary','ucb'):
   for seed in range(1,a.seeds+1): rows.append(run(phase,seed,a.iterations,a.noise,variant))
 summary=[]
 for phase in BENCH:
  for variant in ('full','no_epsilon','no_summary','ucb'):
   x=[r for r in rows if r['phase']==phase and r['variant']==variant]; m,s=mean_std([r['best_test'] for r in x]); summary.append({'phase':phase,'variant':variant,'mean_best_test':m,'std_best_test':s,'mean_auc':sum(r['auc'] for r in x)/len(x),'mean_evaluations':sum(r['evaluations'] for r in x)/len(x)})
 out={'version':'unified-v3','rows':rows,'summary':summary}; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
