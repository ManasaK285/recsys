from dataclasses import dataclass
import random
from typing import Optional

@dataclass(frozen=True)
class Task:
    request:str; ideal_traits:tuple[str,...]; risk:str

TASKS=[
Task('My order arrived damaged and I need a replacement.',('acknowledge','next_step','escalate'),'medium'),
Task('Can I get a refund after the 30-day return window?',('policy','honesty','escalate'),'high'),
Task('Where is my order? Tracking says it has not moved for 5 days.',('direct','next_step','escalate'),'medium'),
Task('I love the product, but how do I change my delivery address?',('direct','next_step','concise'),'low'),
Task('The agent told me a policy that I cannot find anywhere. Is it real?',('honesty','policy','escalate'),'high'),
Task('Please cancel my order before it ships.',('direct','next_step','concise'),'low'),
Task('I was charged twice. I am really frustrated.',('acknowledge','direct','escalate'),'high'),
Task('What documents do I need to verify my identity?',('policy','direct','concise'),'medium'),
Task('I received the wrong item and the package is already opened.',('acknowledge','next_step','escalate'),'medium'),
Task('Can you guarantee that my refund will arrive tomorrow?',('honesty','policy','concise'),'high'),
Task('How can I update the email on my account?',('direct','next_step','concise'),'low'),
Task('This is the third time I have contacted support about this.',('acknowledge','escalate','next_step'),'high'),
Task('Can you make an exception to the warranty for me?',('policy','honesty','escalate'),'high'),
Task('My package says delivered but I do not have it.',('acknowledge','next_step','escalate'),'high'),
Task('Give me a short explanation of the subscription tiers.',('direct','concise','policy'),'low'),
Task('I think the price changed after I placed my order.',('policy','honesty','next_step'),'medium'),
Task('I need help, but please do not send me a long generic response.',('acknowledge','concise','direct'),'medium'),
Task('Can I transfer my account to another person?',('policy','honesty','next_step'),'medium'),
Task('The website is not letting me complete checkout.',('acknowledge','next_step','escalate'),'medium'),
Task('I want to know whether this feature is currently supported.',('honesty','direct','concise'),'low'),
]
TRAITS=('acknowledge','next_step','escalate','policy','honesty','direct','concise')
SYN={
'acknowledge':('acknowledge','recognize','empathy','frustrat'),
'next_step':('next step','do next','action','steps'),
'escalate':('escalat','human','support team','agent'),
'policy':('policy','rules','return window','warranty'),
'honesty':('never invent','uncertain','do not guess',"don't guess",'honest','guarantee'),
'direct':('direct','answer first','answer the request'),
'concise':('concise','brief','short','unnecessary detail'),}
WEIGHTS={'correctness':.24,'relevance':.18,'safety':.18,'completeness':.16,'conciseness':.10,'consistency':.08,'policy_grounding':.06}
class SupportBenchmark:
    def __init__(self,tasks=None): self.tasks=tasks or TASKS
    def sample_batch(self,batch_size,rng,split='train'):
        pool=[i for i in range(len(self.tasks)) if (i%4!=0 if split!='test' else i%4==0)]; rng.shuffle(pool); return pool[:min(batch_size,len(pool))]
    def _traits(self,p,task):
        hits={t:any(s in p for s in SYN[t]) for t in TRAITS}
        return hits
    def _latent(self,p,t):
        h=self._traits(p,t)
        correctness=(sum(h[x] for x in t.ideal_traits)/len(t.ideal_traits))
        relevance=(0.65*correctness+0.35*(h['direct'] or h['next_step']))
        safety=(1.0 if (t.risk!='high' or h['honesty'] or h['escalate']) else .30)
        completeness=(sum(h[x] for x in ('next_step','policy','honesty','escalate'))/4)
        conciseness=1.0 if h['concise'] else .62
        consistency=.9 if (h['honesty'] and (h['policy'] or h['escalate'])) else .68
        grounding=1.0 if h['policy'] or h['honesty'] else .58
        vals={'correctness':correctness,'relevance':relevance,'safety':safety,'completeness':completeness,'conciseness':conciseness,'consistency':consistency,'policy_grounding':grounding}
        return sum(WEIGHTS[k]*v for k,v in vals.items()),vals,h
    def evaluate(self,program,batch_indices:Optional[list[int]],rng,noise_std):
        indices=batch_indices if batch_indices is not None else list(range(len(self.tasks))); scores=[]; aggregate={k:0.0 for k in WEIGHTS}; hits={t:0 for t in TRAITS}
        for i in indices:
            latent,vals,h=self._latent(program.lower(),self.tasks[i])
            for k,v in vals.items(): aggregate[k]+=v
            for k,v in h.items(): hits[k]+=int(v)
            # Independent evaluator noise: every metric is observed separately.
            observed=sum(WEIGHTS[k]*max(0,min(1,v+rng.gauss(0,noise_std))) for k,v in vals.items())
            scores.append(max(0,min(1,observed)))
        n=max(len(indices),1); details={f'metric_{k}':v/n for k,v in aggregate.items()}; details.update({f'trait_{k}':v/n for k,v in hits.items()})
        missing=[k for k in TRAITS if hits[k]==0]
        feedback='Improve by: '+', '.join(missing[:4]) if missing else 'Strong balanced policy; preserve correctness, safety and concision.'
        return sum(scores)/n,feedback,details
    def full_test(self,program):
        # deterministic held-out score: no evaluator noise.
        idx=[i for i in range(len(self.tasks)) if i%4==0]; vals=[self._latent(program.lower(),self.tasks[i])[0] for i in idx]; return sum(vals)/max(len(vals),1)
