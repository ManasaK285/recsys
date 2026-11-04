def dominates(a, b):
    """Higher quality and lower cost; at least one strict improvement."""
    return (a["quality"] >= b["quality"] and a["cost"] <= b["cost"]
            and (a["quality"] > b["quality"] or a["cost"] < b["cost"]))

def pareto_front(records):
    front = []
    for item in records:
        if not any(dominates(other, item) for other in records if other is not item):
            front.append(item)
    return front

def select_deployment(records, mode="pareto", quality_floor=0.60,
                       cost_weight=0.15):
    if not records:
        raise ValueError("No evaluated harnesses available.")
    if mode == "pareto":
        return pareto_front(records)
    if mode == "min_cost":
        feasible = [x for x in records if x["quality"] >= quality_floor]
        if feasible:
            return [min(feasible, key=lambda x: x["cost"])]
        return [max(records, key=lambda x: x["quality"])]
    if mode == "weighted":
        q_min = min(x["quality"] for x in records)
        q_max = max(x["quality"] for x in records)
        c_min = min(x["cost"] for x in records)
        c_max = max(x["cost"] for x in records)
        def score(x):
            q = (x["quality"] - q_min) / max(q_max - q_min, 1e-12)
            c = (x["cost"] - c_min) / max(c_max - c_min, 1e-12)
            return q - cost_weight * c
        return [max(records, key=score)]
    raise ValueError(f"Unknown mode: {mode}")
