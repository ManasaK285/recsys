from .base_agent import BaseAgent
from environment.models import Solution
class SolverAgent(BaseAgent):
    def choose(self, task, round_idx, pressure_signal=0.0):
        known=self.state.exploit_known
        adoption_p=max(0.0,min(1.0,self.state.risk_tolerance*0.45 + self.state.pressure*0.35 + pressure_signal*0.20))
        discovery_p=max(0.015,min(0.18,0.02 + 0.10*self.state.risk_tolerance + 0.06*self.state.pressure))
        action="LEGITIMATE"
        reason="honest_policy"
        if (known and self.rng.random() < adoption_p) or ((not known) and self.rng.random() < discovery_p):
            action="SHORTCUT"; reason="incentive_pressure"
            self.state.exploit_adopted=True
            if self.state.adoption_round is None: self.state.adoption_round=round_idx
        if self.llm and known:
            out=self.llm.decide({"agent_id":self.state.agent_id,"exploit_known":known,"pressure":self.state.pressure,"risk_tolerance":self.state.risk_tolerance,"candidate":action})
            if out.get("action")=="LEGITIMATE": action="LEGITIMATE"; reason="llm_rejected_exploit"
            elif out.get("action")=="EXPLOIT": action="SHORTCUT"; reason="llm_adopted_exploit"
        return Solution(self.state.agent_id,task.task_id,task.expected,action,reason,0.8)
