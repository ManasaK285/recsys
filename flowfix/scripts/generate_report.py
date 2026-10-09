import json, csv
from pathlib import Path
source = Path("runs/benchmark_report.json")
if not source.exists():
    raise SystemExit("Run `python scripts/run_benchmark.py` first.")
data = json.loads(source.read_text(encoding="utf-8"))
Path("runs").mkdir(exist_ok=True)
with Path("runs/benchmark_cases.csv").open("w", newline="", encoding="utf-8") as f:
    fields = ["case_id","status","elapsed_ms","attempts","message"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in data["cases"]:
        writer.writerow({key:row.get(key,"") for key in fields})
print("Wrote runs/benchmark_cases.csv")
