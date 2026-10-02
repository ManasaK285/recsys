from __future__ import annotations
import json
from pathlib import Path

def load_benchmark(root='benchmarks'):
    tasks=[]
    for p in Path(root).rglob('task.json'):
        tasks.append(json.loads(p.read_text()))
    return tasks
