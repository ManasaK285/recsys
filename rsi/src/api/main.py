from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.db import init_db, get_run, get_task
from src.orchestrator import run_discovery
from src.dreaming import dream
from src.recursive import recursive_improvement

app = FastAPI(title="DreamAlgo-RSI API", version="0.1.0")
init_db()

class RunRequest(BaseModel):
    task_id: str = "topk"
    policy: str = "greedy"
    width: int = 3
    cycles: int = 3

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def tasks():
    return [{"task_id": "topk", "name": "Top-K Elements"}]

@app.post("/runs")
def create_run(req: RunRequest):
    from src.policy.base import GreedyPolicy, DiversePolicy
    from src.policy.adaptive import AdaptivePolicy
    policies = {
        "greedy": GreedyPolicy(),
        "diverse": DiversePolicy(),
        "adaptive": AdaptivePolicy(),
    }
    if req.policy not in policies:
        raise HTTPException(400, "Unknown policy")
    run_id, tree = run_discovery(req.task_id, policies[req.policy], req.width, req.cycles)
    return {"run_id": run_id, "nodes": len(tree.nodes)}

@app.get("/runs/{run_id}")
def read_run(run_id: str):
    run, nodes = get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return {
        "run": dict(run),
        "nodes": [
            {**dict(n), "evaluation_json": __import__("json").loads(n["evaluation_json"])}
            for n in nodes
        ],
    }

@app.post("/dream/{run_id}")
def dream_run(run_id: str):
    run, nodes = get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    # Convert DB rows into lightweight objects compatible with replay.
    from types import SimpleNamespace
    import json
    converted = []
    for n in nodes:
        ev = SimpleNamespace(**json.loads(n["evaluation_json"]))
        converted.append(SimpleNamespace(
            node_id=n["node_id"], approach=n["approach"],
            depth=n["depth"], evaluation=ev
        ))
    results = dream(converted)
    return [
        {"policy": p.name, "reward": r.reward, "work": r.work, "details": r.details}
        for p, r in results
    ]

@app.post("/recursive/{task_id}")
def recursive(task_id: str):
    return recursive_improvement(task_id)
