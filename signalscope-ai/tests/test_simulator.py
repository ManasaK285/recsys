from signalscope.simulator import generate_telemetry, AFFECTED_SERVICES

def test_seeded_data_has_injected_anomaly():
    df = generate_telemetry(days=7, service_count=5, seed=7)
    truth = df[df.is_injected_anomaly]
    assert not truth.empty
    assert set(truth.service.unique()).issubset(set(AFFECTED_SERVICES))
    assert set(truth.region.unique()) == {"us-east"}

def test_same_seed_reproduces_metrics():
    a = generate_telemetry(days=7, service_count=5, seed=17)
    b = generate_telemetry(days=7, service_count=5, seed=17)
    assert a.latency_p95_ms.tolist() == b.latency_p95_ms.tolist()
