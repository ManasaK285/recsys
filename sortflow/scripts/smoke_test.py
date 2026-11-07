import requests, time, sys

base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
assert requests.get(base + "/health", timeout=5).ok
event = requests.post(base + "/events", json={
    "event_type":"DEMAND_SURGE", "center_id":"SC-08",
    "volume_change_pct":40, "severity":"HIGH"
}, timeout=5)
event.raise_for_status()
workflow_id = event.json()["workflow_id"]
print("Queued:", workflow_id)
for _ in range(30):
    detail = requests.get(base + f"/workflows/{workflow_id}", timeout=5).json()
    print("Status:", detail["status"])
    if detail["status"] in ("COMPLETED", "FAILED"):
        print(detail)
        if detail["status"] != "COMPLETED":
            raise SystemExit(1)
        break
    time.sleep(1)
else:
    raise SystemExit("Workflow did not finish in 30 seconds")
