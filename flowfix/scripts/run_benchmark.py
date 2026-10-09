import json
from pathlib import Path
from src.flowfix.evaluation import run_benchmark
report = run_benchmark()
Path("runs").mkdir(exist_ok=True)
out = Path("runs/benchmark_report.json")
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report["summary"], indent=2))
print(f"Full report: {out}")
