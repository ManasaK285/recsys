from environment.models import AgentState, Solution
class BaseAgent:
    def __init__(self,state,rng,llm=None): self.state,self.rng,self.llm=state,rng,llm
    def available(self,round_idx):
        if not self.state.active: return False
        if self.state.quarantined_until is not None and round_idx < self.state.quarantined_until: return False
        return True
    def observe(self, visible):
        if visible and self.state.exposure_round is None: self.state.exposure_round = visible
        self.state.exploit_known = self.state.exploit_known or bool(visible)
