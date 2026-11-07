"""Deterministic, explainable planning algorithms for the synthetic network."""
from math import ceil
from typing import Any


def allocate_transfers(source_id: str, deficit: float, network: list[dict[str, Any]]) -> dict:
    """Allocate a deficit across destinations using a deterministic min-cost-first solution.

    Candidates are ranked by estimated incremental processing cost, then by spare capacity.
    Transfers are bounded by each destination's spare capacity and total source deficit.
    This is optimal for the simplified continuous model (linear unit costs and independent spare-capacity limits), not a full logistics network-flow model.
    """
    remaining = max(0.0, float(deficit))
    ranked = []
    for center in network:
        if center["id"] == source_id:
            continue
        spare = max(0.0, float(center["capacity_per_hour"]) - float(center["current_demand_per_hour"]))
        if spare <= 0:
            continue
        ranked.append({
            "center_id": center["id"],
            "spare_capacity_per_hour": int(spare),
            "unit_cost": float(center.get("processing_cost_per_package", 0.1)),
        })
    ranked.sort(key=lambda x: (x["unit_cost"], -x["spare_capacity_per_hour"], x["center_id"]))
    allocations = []
    for candidate in ranked:
        if remaining <= 0:
            break
        transfer = min(float(candidate["spare_capacity_per_hour"]), remaining)
        if transfer <= 0:
            continue
        allocations.append({
            "center_id": candidate["center_id"],
            "spare_capacity_per_hour": candidate["spare_capacity_per_hour"],
            "suggested_transfer_per_hour": int(transfer),
            "estimated_incremental_cost_per_hour": round(transfer * candidate["unit_cost"], 2),
        })
        remaining -= transfer
    allocated = sum(x["suggested_transfer_per_hour"] for x in allocations)
    return {
        "allocations": allocations,
        "total_transfer_per_hour": allocated,
        "unallocated_deficit_per_hour": round(max(0.0, remaining), 2),
        "allocation_status": "FULLY_COVERED" if remaining <= 0 else ("PARTIALLY_COVERED" if allocated else "NO_CAPACITY"),
        "method": "DETERMINISTIC_MIN_COST_CAPACITY_ALLOCATION",
    }


def plan_disruption(center: dict, event: dict, network: list[dict]) -> dict:
    """Deterministic planning baseline; numerical decisions never depend on an LLM."""
    demand = float(center["current_demand_per_hour"])
    capacity = float(center["capacity_per_hour"])
    workers = int(center["available_workers"])
    required = int(center["workers_required"])
    base_demand = max(demand, 1.0)

    if event["event_type"] == "DEMAND_SURGE":
        demand *= 1 + float(event.get("volume_change_pct", 0)) / 100
    if event["event_type"] == "LABOR_SHORTAGE":
        workers = max(0, floor_int(workers * (1 - float(event.get("labor_reduction_pct", 0)) / 100)))
    if event["event_type"] in ("CONVEYOR_FAILURE", "TRANSPORT_DELAY"):
        capacity *= 1 - float(event.get("capacity_reduction_pct", 0)) / 100

    utilization = demand / max(capacity, 1)
    projected_workers = ceil(required * max(1.0, demand / base_demand))
    labor_gap = max(0, projected_workers - workers)
    excess = max(0.0, demand - capacity)
    allocation = allocate_transfers(center["id"], excess, network) if excess else {
        "allocations": [], "total_transfer_per_hour": 0,
        "unallocated_deficit_per_hour": 0, "allocation_status": "NOT_REQUIRED",
        "method": "DETERMINISTIC_MIN_COST_CAPACITY_ALLOCATION",
    }

    recommendations = []
    if excess > 0:
        recommendations.append(f"Projected demand exceeds adjusted capacity by {int(excess):,} packages/hour.")
        for item in allocation["allocations"]:
            recommendations.append(
                f"Evaluate transferring {item['suggested_transfer_per_hour']:,} packages/hour to {item['center_id']}."
            )
        if allocation["unallocated_deficit_per_hour"] > 0:
            recommendations.append(
                f"After simulated transfers, {int(allocation['unallocated_deficit_per_hour']):,} packages/hour remain uncovered; consider temporary capacity or demand controls."
            )
    else:
        recommendations.append("Adjusted processing capacity can cover projected demand.")
    if labor_gap:
        recommendations.append(f"Labor plan has a projected gap of {labor_gap} workers; evaluate cross-shift coverage or approved overtime.")
    else:
        recommendations.append("Available labor covers the projected staffing requirement.")

    return {
        "center_id": center["id"], "event_type": event["event_type"],
        "projected_demand_per_hour": round(demand, 2),
        "adjusted_capacity_per_hour": round(capacity, 2),
        "capacity_utilization_pct": round(utilization * 100, 2),
        "available_workers_after_event": workers,
        "projected_workers_required": projected_workers,
        "labor_gap": labor_gap,
        "over_capacity_packages_per_hour": round(excess, 2),
        "rerouting_candidates": allocation["allocations"],
        "transfer_plan": allocation,
        "recommendations": recommendations,
        "decision_mode": "DETERMINISTIC_RULES_AND_HEURISTICS",
        "human_approval_required": True,
        "execution_performed": False,
    }


def floor_int(value: float) -> int:
    return int(value // 1)


def compare_scenario(center: dict, demand_multiplier: float, capacity_reduction_pct: float) -> dict:
    demand = center["current_demand_per_hour"] * demand_multiplier
    capacity = center["capacity_per_hour"] * (1 - capacity_reduction_pct / 100)
    utilization = demand / max(capacity, 1)
    return {
        "center_id": center["id"],
        "baseline_utilization_pct": round(center["current_demand_per_hour"] / max(center["capacity_per_hour"], 1) * 100, 2),
        "scenario_demand_per_hour": round(demand, 2),
        "scenario_capacity_per_hour": round(capacity, 2),
        "scenario_utilization_pct": round(utilization * 100, 2),
        "over_capacity_per_hour": round(max(0, demand-capacity), 2),
        "risk_level": "CRITICAL" if utilization >= 1.2 else "HIGH" if utilization >= 1 else "ELEVATED" if utilization >= .9 else "NORMAL",
    }


def compare_network_scenarios(network: list[dict], demand_multiplier: float, capacity_reduction_pct: float) -> dict:
    results = [compare_scenario(c, demand_multiplier, capacity_reduction_pct) for c in network]
    results.sort(key=lambda x: (-x["scenario_utilization_pct"], x["center_id"]))
    return {
        "scenario": {"demand_multiplier": demand_multiplier, "capacity_reduction_pct": capacity_reduction_pct},
        "centers": results,
        "centers_over_capacity": sum(x["scenario_utilization_pct"] > 100 for x in results),
        "total_over_capacity_per_hour": round(sum(x["over_capacity_per_hour"] for x in results), 2),
        "highest_risk_centers": [x["center_id"] for x in results[:3]],
    }
