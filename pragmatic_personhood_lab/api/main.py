from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any
from src.database import init_db, seed_agents, list_agents, upsert_agent, get_agent
from src.governance.service import evaluate_and_record, set_agent_status
from src.audit.event_log import list_events

app = FastAPI(title="Pragmatic Personhood Lab API", version="1.0.0")
init_db()
seed_agents()

class AgentInput(BaseModel):
    agent_id: str
    name: str
    role: str = "custom"
    owner: str
    permissions: list[str] = Field(default_factory=list)
    purchase_limit: float = 0
    status: str = "active"
    require_human_approval_for_external: bool = True

class ActionInput(BaseModel):
    agent_id: str
    action_type: str
    resource: str = "unspecified"
    amount: float = 0
    external: bool = False
    human_approval: bool = False
    context: dict[str, Any] = Field(default_factory=dict)

@app.get("/health")
def health():
    return {"status":"ok","service":"pragmatic-personhood-lab"}

@app.get("/agents")
def agents():
    return list_agents()

@app.post("/agents")
def create_agent(agent: AgentInput):
    upsert_agent(agent.model_dump())
    return get_agent(agent.agent_id)

@app.post("/evaluate")
def evaluate(action: ActionInput):
    return evaluate_and_record(action.model_dump())

@app.get("/audit")
def audit(limit: int = 100):
    return list_events(max(1, min(limit, 1000)))

@app.post("/agents/{agent_id}/suspend")
def suspend(agent_id: str):
    try:
        return set_agent_status(agent_id, "suspended")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

@app.post("/agents/{agent_id}/reinstate")
def reinstate(agent_id: str):
    try:
        return set_agent_status(agent_id, "active")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
