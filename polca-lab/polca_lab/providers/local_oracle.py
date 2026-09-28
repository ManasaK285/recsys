import random
from typing import List

MUTATIONS = [
    "Be concise and answer the request directly.",
    "Acknowledge the customer's issue briefly before solving it.",
    "Give a concrete next step whenever possible.",
    "Never invent policy, guarantees, or unavailable information.",
    "State uncertainty clearly and recommend human review when needed.",
    "Escalate high-risk or policy-sensitive cases to human support.",
    "Avoid generic filler and unnecessary background.",
]

class LocalProposalOracle:
    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)

    def propose(self, parent: str, feedback: str, summary: str, n: int = 3) -> List[str]:
        proposals = []
        for _ in range(n):
            lines = [x.strip() for x in parent.splitlines() if x.strip()]
            mutation = self.rng.choice(MUTATIONS)
            if mutation.lower() not in {x.lower() for x in lines}:
                lines.append(mutation)
            if self.rng.random() < 0.35 and len(lines) > 3:
                # Small compression mutation creates a useful exploration branch.
                lines.pop(self.rng.randrange(len(lines)))
            proposals.append(" ".join(lines))
        return list(dict.fromkeys(proposals))
