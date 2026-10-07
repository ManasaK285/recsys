from __future__ import annotations
import asyncio, json
from pathlib import Path
from harness.orchestrator import AgentForge
from evaluation.benchmark import load_benchmark

def main():
    tasks=load_benchmark('benchmarks')
    results=[]
    for task in tasks:
        result=asyncio.run(AgentForge().run(task['task'],task.get('repository','sample_repo'),3))
        results.append({'task_id':task['id'],'status':result['status'],'attempts':result['attempts']})
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/benchmark_results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
