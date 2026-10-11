import sys, json, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from provenancelab.benchmark import run_benchmark

out_dir = ROOT / "outputs"
out_dir.mkdir(exist_ok=True)
result = run_benchmark(samples_per_class=30, seed=42)
(out_dir / "benchmark_summary.json").write_text(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2), encoding="utf-8")
with (out_dir / "benchmark_rows.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=result["rows"][0].keys())
    writer.writeheader(); writer.writerows(result["rows"])
print(json.dumps(result["metrics"], indent=2))
print(result["warning"])
