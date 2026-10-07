import argparse, json, csv
from pathlib import Path
from polca_lab.core.optimizer import POLCAOptimizer
from polca_lab.core.evaluator import StochasticEvaluator
from polca_lab.providers.unified_oracle import UnifiedLocalOracle
from polca_lab.benchmarks import SupportBenchmark,RAGBenchmark,AgentBenchmark

SEEDS={
 'prompt':['Answer customer requests directly and helpfully.'],
 'rag':['Answer questions helpfully using retrieved context.'],
 'agent':['Handle customer requests safely and helpfully. Use lookup_order before actions that depend on order state. Verify identity before sensitive actions.'],
}
BENCH={'prompt':SupportBenchmark,'rag':RAGBenchmark,'agent':AgentBenchmark}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--phase',choices=BENCH,default='prompt'); p.add_argument('--iterations',type=int,default=35); p.add_argument('--seed',type=int,default=7); p.add_argument('--out',default=None); p.add_argument('--no-epsilon',action='store_true'); p.add_argument('--no-summary',action='store_true'); p.add_argument('--priority',choices=['mean','ucb'],default='mean'); a=p.parse_args()
 bench=BENCH[a.phase](); cfg={'seed':a.seed,'memory_size':24,'priority':a.priority,'epsilon':.28,'use_epsilon_net':not a.no_epsilon,'use_summary':not a.no_summary,'iterations':a.iterations,'proposals_per_iteration':3,'batch_size':8,'noise_std':.08}
 ev=StochasticEvaluator(bench,noise_std=cfg['noise_std'],seed=a.seed); opt=POLCAOptimizer(ev,UnifiedLocalOracle(a.phase,a.seed),cfg); opt.initialize(SEEDS[a.phase]); log=opt.run(a.iterations)
 out=Path(a.out or f'results/{a.phase}_phase.json'); out.parent.mkdir(exist_ok=True,parents=True); opt.export(str(out)); print(json.dumps({'phase':a.phase,'evaluations':ev.total_evals,'best_test':log[-1]['best_test_score']},indent=2))
if __name__=='__main__': main()
