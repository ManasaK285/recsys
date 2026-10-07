from __future__ import annotations
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from harness.orchestrator import AgentForge
from storage.db import Database

app=FastAPI(title='AgentForge API', version='0.1.0')
engine=AgentForge()
db=Database()

class RunRequest(BaseModel):
    task: str
    repository: str='sample_repo'
    parallel_agents: int=Field(default=3,ge=1,le=3)

@app.get('/health')
def health(): return {'status':'ok'}

@app.post('/runs')
def create_run(req: RunRequest):
    return asyncio.run(engine.run(req.task,req.repository,req.parallel_agents))

@app.get('/runs/{run_id}')
def get_run(run_id: str):
    result=db.get_run(run_id)
    if not result: raise HTTPException(404,'Run not found')
    return result
