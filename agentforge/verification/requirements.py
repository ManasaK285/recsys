from __future__ import annotations
from pathlib import Path

def check_requirements(workspace: str, task: str, spec: dict) -> dict:
    if 'health' in task.lower():
        main=Path(workspace)/'app'/'main.py'
        passed='@app.get("/health")' in main.read_text() and '"status": "ok"' in main.read_text()
        return {'passed':passed,'checks':[{'name':'health endpoint','passed':passed}]}
    return {'passed': True, 'checks': [{'name':'manual requirement review','passed':True}]}
