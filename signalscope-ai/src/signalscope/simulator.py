from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

AFFECTED_SERVICES = ["checkout", "search", "recommendations"]

def generate_telemetry(days=30, service_count=50, regions=("us-east", "us-west", "eu-west"), seed=42, interval_minutes=60):
    if days < 7:
        raise ValueError("days must be at least 7")
    if service_count < 3:
        raise ValueError("service_count must be at least 3")
    if interval_minutes not in (5, 15, 30, 60):
        raise ValueError("interval_minutes must be 5, 15, 30, or 60")
    rng = np.random.default_rng(seed)
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    stamps = pd.date_range(start=start, periods=days * 24 * 60 // interval_minutes, freq=f"{interval_minutes}min", tz="UTC")
    services = AFFECTED_SERVICES + [f"service-{i:02d}" for i in range(service_count - 3)]
    release_start = start + timedelta(days=days - 5)
    incident_start = start + timedelta(days=days - 3)
    rows = []
    for idx, service in enumerate(services):
        base_latency = rng.uniform(45, 180)
        base_error = rng.uniform(.001, .009)
        base_cpu = rng.uniform(20, 58)
        base_mem = rng.uniform(30, 70)
        base_requests = int(rng.integers(250, 1800))
        for ridx, region in enumerate(regions):
            region_factor = 1 + ridx * .08
            for ts in stamps:
                affected = service in AFFECTED_SERVICES and region == "us-east" and ts.to_pydatetime() >= incident_start
                release = "v2.4.1" if ts.to_pydatetime() >= release_start else "v2.4.0"
                cycle = 1 + .1 * np.sin(ts.hour / 24 * 2 * np.pi)
                requests = max(0, int(base_requests * cycle + rng.normal(0, base_requests * .07)))
                scale = 1.48 if affected else 1.0
                p50 = max(1, base_latency * region_factor * scale + rng.normal(0, 5))
                p95 = max(p50, base_latency * region_factor * 1.75 * scale + rng.normal(0, 10))
                error = np.clip(base_error + rng.normal(0, .0015) + (.018 if affected else 0), 0, 1)
                cpu = np.clip(base_cpu + 12 * requests / max(base_requests, 1) * .01 + rng.normal(0, 4) + (8 if affected else 0), 0, 100)
                memory = np.clip(base_mem + rng.normal(0, 3) + (4 if affected else 0), 0, 100)
                queue = max(0, int(rng.poisson(3) + (18 if affected else 0)))
                rows.append({
                    "timestamp": ts, "service": service, "region": region, "release": release,
                    "latency_p50_ms": round(float(p50), 3), "latency_p95_ms": round(float(p95), 3),
                    "error_rate": round(float(error), 6), "cpu_pct": round(float(cpu), 3),
                    "memory_pct": round(float(memory), 3), "request_count": requests,
                    "queue_depth": queue, "is_injected_anomaly": bool(affected)
                })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
