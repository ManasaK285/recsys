import csv
import math
import os
import statistics


def _numeric_values(rows, key):
    return [
        r[key]
        for r in rows
        if isinstance(r.get(key), (int, float))
        and not isinstance(r.get(key), bool)
    ]


def aggregate(rows):
    if not rows:
        return {}

    numeric_keys = [
        k
        for k in rows[0]
        if k != "seed"
        and k != "governance"
        and k != "evaluator"
        and isinstance(rows[0].get(k), (int, float))
        and not isinstance(rows[0].get(k), bool)
    ]

    result = {
        "governance": rows[0]["governance"],
        "evaluator": rows[0].get("evaluator", "unknown"),
        "runs": len(rows),
    }

    for key in numeric_keys:
        values = _numeric_values(rows, key)
        if not values:
            result[key] = None
            continue

        result[key] = statistics.mean(values)

        if len(values) > 1:
            result[f"{key}_std"] = statistics.stdev(values)
        else:
            result[f"{key}_std"] = 0.0

    return result


def write_csv(rows, path):
    if not rows:
        return

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)