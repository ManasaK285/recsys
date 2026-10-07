import json, random, time
from pathlib import Path
from .candidate import Candidate
from .diversity import EpsilonNet
from .memory import CandidateMemory
from .summarizer import LocalSummarizer

class POLCAOptimizer:
    def __init__(self,evaluator,oracle,config):
        self.evaluator=evaluator; self.oracle=oracle; self.config=config
        self.rng=random.Random(config.get('seed',7))
        self.memory=CandidateMemory(config.get('memory_size',24),config.get('priority','mean'))
        self.net=EpsilonNet(config.get('epsilon',0.28)); self.summarizer=LocalSummarizer()
        self.use_net=config.get('use_epsilon_net',True); self.use_summary=config.get('use_summary',True)
        self.iteration_log=[]; self.cid_counter=0
    def _new_id(self): self.cid_counter+=1; return f'c{self.cid_counter:04d}'
    def initialize(self,seed_programs):
        for p in seed_programs: self.memory.add(Candidate(self._new_id(),p,created_at=0))
        self._evaluate_candidates(self.memory.select(min(len(seed_programs),2)),None)
    def _evaluate_candidates(self,candidates,batch):
        for c in candidates:
            ev=self.evaluator.evaluate(c.program,batch); self.memory.record_eval(c.cid,ev.score,ev.feedback)
    def run(self,iterations=None):
        iterations=iterations or self.config.get('iterations',35); pp=self.config.get('proposals_per_iteration',3); bs=self.config.get('batch_size',12); start=time.time()
        for it in range(1,iterations+1):
            batch=self.evaluator.benchmark.sample_batch(bs,self.rng)
            selected=self.memory.select(min(2,len(self.memory.items))); self._evaluate_candidates(selected,batch)
            summary=self.summarizer.summarize(list(self.memory.items.values())) if self.use_summary else 'No global summary is available.'
            accepted=rejected=generated=0
            for parent in selected:
                for program in self.oracle.propose(parent.program,parent.latest_feedback,summary,pp):
                    generated+=1
                    if any(c.program.strip().lower()==program.strip().lower() for c in self.memory.items.values()): rejected+=1; continue
                    if self.use_net and not self.net.accept(program,self.memory.texts()): rejected+=1; self.memory.rejections+=1; continue
                    c=Candidate(self._new_id(),program,created_at=it); ev=self.evaluator.evaluate(program,batch); c.record(ev.score,ev.feedback); self.memory.add(c); accepted+=1
            best=self.memory.best(); test=self.evaluator.benchmark.full_test(best.program) if best else 0.0
            self.iteration_log.append({'iteration':it,'evaluations':self.evaluator.total_evals,'generated':generated,'accepted':accepted,'rejected':rejected,'memory_size':len(self.memory.items),'best_train_mean':best.mean if best else 0.0,'best_test_score':test,'elapsed_sec':time.time()-start})
        return self.iteration_log
    def export(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps({'config':self.config,'total_evals':self.evaluator.total_evals,'rejections':self.memory.rejections,'iterations':self.iteration_log,'candidates':[c.to_dict() for c in self.memory.items.values()]},indent=2),encoding='utf-8')
