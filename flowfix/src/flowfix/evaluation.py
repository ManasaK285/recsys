import statistics
from .agent import repair_case
from .benchmarks import CASES

def run_benchmark():
    rows = [repair_case(case_id) for case_id in CASES]
    total = len(rows)
    latencies = sorted(row["elapsed_ms"] for row in rows)
    p95 = latencies[min(total - 1, int(0.95 * total))] if total else 0
    return {"cases": rows, "summary": {
        "total_cases": total,
        "verified_suggestions": sum(r["status"] == "verified_suggestion" for r in rows),
        "verified_success_rate": sum(r["status"] == "verified_suggestion" for r in rows) / total if total else 0,
        "abstentions": sum(r["status"] == "abstained" for r in rows),
        "median_latency_ms": statistics.median(latencies) if latencies else 0,
        "p95_latency_ms": p95,
        "note": "Four seeded demo cases only; not comparable to industrial or research benchmarks."
    }}
