import os
import requests
from mcp.server.fastmcp import FastMCP

API = os.getenv("API_BASE_URL", "http://localhost:8000")
mcp = FastMCP("SortFlow Operations Tools")

def api_get(path):
    r = requests.get(f"{API}{path}", timeout=10)
    r.raise_for_status()
    return r.json()

@mcp.tool()
def get_network_summary() -> dict:
    """Return network-wide demand, capacity, utilization and workflow KPIs."""
    return api_get("/summary")

@mcp.tool()
def get_center_status(center_id: str) -> dict:
    """Return current simulated capacity, demand, labor and utilization for a sort center."""
    centers = api_get("/centers")
    for center in centers:
        if center["id"] == center_id:
            return center
    return {"error": f"Unknown center: {center_id}"}

@mcp.tool()
def get_recent_events(limit: int = 10) -> list:
    """List recent network disruptions and processing status."""
    return api_get(f"/events?limit={max(1, min(limit, 100))}")

@mcp.tool()
def get_workflow_status(workflow_id: str) -> dict:
    """Inspect a workflow and its persisted planning result."""
    return api_get(f"/workflows/{workflow_id}")

@mcp.tool()
def compare_planning_scenarios(center_id: str, demand_multiplier: float = 1.2,
                               capacity_reduction_pct: float = 0) -> dict:
    """Compare a demand/capacity scenario for a center; this does not execute an action."""
    r = requests.post(f"{API}/scenarios/compare", json={
        "center_id": center_id,
        "demand_multiplier": demand_multiplier,
        "capacity_reduction_pct": capacity_reduction_pct
    }, timeout=10)
    r.raise_for_status()
    return r.json()

@mcp.tool()
def propose_operational_action(center_id: str, action: str, rationale: str) -> dict:
    """Create a text-only proposed action for human review; never executes real operations."""
    return {
        "status": "PROPOSED_FOR_HUMAN_REVIEW",
        "center_id": center_id,
        "action": action,
        "rationale": rationale,
        "execution_performed": False
    }


@mcp.tool()
def compare_network_scenarios(demand_multiplier: float = 1.2, capacity_reduction_pct: float = 0) -> dict:
    """Rank all simulated centers under a network-wide demand/capacity scenario."""
    r = requests.post(f"{API}/scenarios/network", json={
        "demand_multiplier": demand_multiplier,
        "capacity_reduction_pct": capacity_reduction_pct
    }, timeout=15)
    r.raise_for_status()
    return r.json()

@mcp.tool()
def get_audit_history(entity_id: str = "", limit: int = 50) -> list:
    """Read audit history for an entity, or recent network audit records."""
    suffix = f"?entity_id={entity_id}&limit={max(1, min(limit, 200))}" if entity_id else f"?limit={max(1, min(limit, 200))}"
    return api_get(f"/audit{suffix}")

if __name__ == "__main__":
    mcp.run()
