from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path

class TraceLogger:
    def __init__(self, run_dir: str):
        self.path=Path(run_dir)/'trajectory.jsonl'; self.path.parent.mkdir(parents=True,exist_ok=True)
    def log(self, event: str, **data):
        row={'timestamp':datetime.now(timezone.utc).isoformat(),'event':event,**data}
        with self.path.open('a',encoding='utf-8') as f: f.write(json.dumps(row,default=str)+'\n')
