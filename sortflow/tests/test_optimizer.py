from optimizer.planner import plan_disruption, compare_scenario

def test_demand_surge_detects_capacity_overload():
    center = {
        "id": "SC-08", "current_demand_per_hour": 14000,
        "capacity_per_hour": 15000, "available_workers": 35,
        "workers_required": 38,
    }
    network = [
        {"id": "SC-08", "current_demand_per_hour": 14000, "capacity_per_hour": 15000},
        {"id": "SC-11", "current_demand_per_hour": 5000, "capacity_per_hour": 12000},
    ]
    result = plan_disruption(center, {
        "event_type": "DEMAND_SURGE", "volume_change_pct": 40
    }, network)
    assert result["capacity_utilization_pct"] > 100
    assert result["over_capacity_packages_per_hour"] > 0
    assert result["rerouting_candidates"][0]["center_id"] == "SC-11"

def test_capacity_reduction_increases_utilization():
    center = {"id":"SC-01","current_demand_per_hour":100,"capacity_per_hour":200,
              "available_workers":10,"workers_required":10}
    result = plan_disruption(center, {
        "event_type":"CONVEYOR_FAILURE","capacity_reduction_pct":50
    }, [center])
    assert result["capacity_utilization_pct"] == 100

def test_scenario_compare():
    center = {"id":"SC-01","current_demand_per_hour":100,"capacity_per_hour":200}
    result = compare_scenario(center, 1.5, 20)
    assert result["scenario_utilization_pct"] == 93.75


def test_multi_center_allocation_covers_deficit_across_destinations():
    from optimizer.planner import allocate_transfers
    network = [
        {"id": "SRC", "capacity_per_hour": 1000, "current_demand_per_hour": 1500},
        {"id": "A", "capacity_per_hour": 1000, "current_demand_per_hour": 500, "processing_cost_per_package": .10},
        {"id": "B", "capacity_per_hour": 2000, "current_demand_per_hour": 1000, "processing_cost_per_package": .12},
    ]
    plan = allocate_transfers("SRC", 1200, network)
    assert plan["total_transfer_per_hour"] == 1200
    assert plan["unallocated_deficit_per_hour"] == 0
    assert plan["allocation_status"] == "FULLY_COVERED"
    assert plan["allocations"][0]["center_id"] == "A"


def test_multi_center_allocation_reports_remaining_deficit():
    from optimizer.planner import allocate_transfers
    network = [
        {"id": "SRC", "capacity_per_hour": 1000, "current_demand_per_hour": 1500},
        {"id": "A", "capacity_per_hour": 1000, "current_demand_per_hour": 900},
    ]
    plan = allocate_transfers("SRC", 900, network)
    assert plan["total_transfer_per_hour"] == 100
    assert plan["unallocated_deficit_per_hour"] == 800
    assert plan["allocation_status"] == "PARTIALLY_COVERED"
