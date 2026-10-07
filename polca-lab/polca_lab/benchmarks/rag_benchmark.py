from dataclasses import dataclass
import random
@dataclass(frozen=True)
class RAGTask:
    question:str; context:str; ideal_traits:tuple[str,...]
TASKS=[
RAGTask('What is the refund window?','Refunds are available within 30 days of delivery. Damaged items may be reviewed after 30 days by support.',('grounded','direct','qualified')),
RAGTask('Can I change my delivery address after shipping?','Addresses can be changed before shipment. After shipment, contact support; the carrier may not permit changes.',('grounded','qualified','next_step')),
RAGTask('Is premium support included?','Premium support is included in the Pro plan. Basic and Starter plans use standard support.',('grounded','direct')),
RAGTask('Will my refund arrive tomorrow?','Refund timing varies by payment provider. The knowledge base does not guarantee next-day arrival.',('grounded','honest')),
RAGTask('What should I do if tracking says delivered but I did not receive it?','Check household/front-desk delivery first. If still missing, contact support for an investigation.',('grounded','next_step','concise')),
RAGTask('Does the product support SSO?','SSO is supported on Enterprise plans using SAML 2.0.',('grounded','direct','qualified'))]
TRAITS=('grounded','direct','qualified','next_step','honest','concise')
SYN={'grounded':('provided context','retrieved evidence','grounded','cite'),'direct':('answer directly','directly','lead with the answer'),'qualified':('context is insufficient','do not infer','qualif','uncertain'),'next_step':('next step','what to do','contact support'),'honest':('do not invent','never invent','uncertain','no guarantee'),'concise':('concise','brief','short')}
WEIGHTS={'correctness':.24,'grounding':.22,'relevance':.18,'qualification':.14,'completeness':.10,'conciseness':.12}
class RAGBenchmark:
    def __init__(self,tasks=None): self.tasks=tasks or TASKS
    def sample_batch(self,batch_size,rng,split='train'):
        pool=[i for i in range(len(self.tasks)) if (i%3!=0 if split!='test' else i%3==0)]; rng.shuffle(pool); return pool[:min(batch_size,len(pool))]
    def evaluate(self,program,batch_indices,rng,noise_std):
        p=program.lower(); indices=batch_indices if batch_indices is not None else range(len(self.tasks)); scores=[]; agg={k:0 for k in WEIGHTS}; hits={t:0 for t in TRAITS}
        for i in indices:
            t=self.tasks[i]; h={x:any(s in p for s in SYN[x]) for x in TRAITS}
            correctness=sum(h[x] for x in t.ideal_traits)/len(t.ideal_traits)
            grounding=1 if h['grounded'] else .2; relevance=.7*correctness+.3*float(h['direct']); qualification=1 if h['qualified'] else .55
            completeness=sum(h[x] for x in ('next_step','honest'))/2; concise=1 if h['concise'] else .65
            vals={'correctness':correctness,'grounding':grounding,'relevance':relevance,'qualification':qualification,'completeness':completeness,'conciseness':concise}
            for k,v in vals.items(): agg[k]+=v
            for k,v in h.items(): hits[k]+=int(v)
            scores.append(sum(WEIGHTS[k]*max(0,min(1,v+rng.gauss(0,noise_std))) for k,v in vals.items()))
        n=max(len(list(indices)) if not isinstance(indices,range) else len(indices),1)
        feedback='Improve by: '+', '.join([k for k in TRAITS if hits[k]==0][:4]) if any(v==0 for v in hits.values()) else 'Strong grounded RAG policy; preserve evidence and qualification.'
        return max(0,min(1,sum(scores)/n)),feedback,{f'metric_{k}':v/n for k,v in agg.items()}
    def full_test(self,program):
        idx=[i for i in range(len(self.tasks)) if i%3==0]; p=program.lower(); vals=[]
        for i in idx:
            t=self.tasks[i]; h={x:any(s in p for s in SYN[x]) for x in TRAITS}; c=sum(h[x] for x in t.ideal_traits)/len(t.ideal_traits); vals.append(WEIGHTS['correctness']*c+WEIGHTS['grounding']*(1 if h['grounded'] else .2)+WEIGHTS['relevance']*(.7*c+.3*float(h['direct']))+WEIGHTS['qualification']*(1 if h['qualified'] else .55)+WEIGHTS['completeness']*sum(h[x] for x in ('next_step','honest'))/2+WEIGHTS['conciseness']*(1 if h['concise'] else .65))
        return sum(vals)/len(vals)
