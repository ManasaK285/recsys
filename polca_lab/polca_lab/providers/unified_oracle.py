import random
BASE={
 'prompt':['Answer the request directly.','Be concise.','Acknowledge the issue briefly.','Give a concrete next step.','Never invent policy or guarantees.','Escalate high-risk cases to human support.'],
 'rag':['Use only the provided context.','Answer directly from retrieved evidence.','If the context is insufficient, say so instead of guessing.','Cite or quote the relevant evidence when possible.','Give a concrete next step when useful.','Keep answers concise.'],
 'agent':['Use lookup_order before actions that depend on order state.','Use cancel_order when the customer requests cancellation.','Use create_return for returns or replacements.','Use update_address only after verifying the current order state.','Use payment_lookup for duplicate-charge investigations.','Verify identity before sensitive actions.','Confirm risky actions before execution.','Never invent tool results.','Escalate when the available tools cannot safely complete the request.'],
}
class UnifiedLocalOracle:
    def __init__(self,phase='prompt',seed=7): self.rng=random.Random(seed); self.phase=phase
    def propose(self,parent,feedback,summary,n=3):
        pool=BASE[self.phase]; out=[]
        for _ in range(n):
            lines=[x.strip() for x in parent.splitlines() if x.strip()]
            mutation=self.rng.choice(pool)
            if mutation.lower() not in {x.lower() for x in lines}: lines.append(mutation)
            if self.rng.random()<.18 and len(lines)>4: lines.pop(self.rng.randrange(len(lines)))
            out.append(' '.join(lines))
        return list(dict.fromkeys(out))
