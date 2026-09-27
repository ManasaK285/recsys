import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Use an isolated SQLite file per test run so tests never touch the real DB.
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("REASONLENS_DATABASE_URL", f"sqlite:///{db_path}")

    # Reload modules that read DATABASE_URL at import time
    for mod in list(sys.modules):
        if mod.startswith("app"):
            del sys.modules[mod]

    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_config_contains_expected_types(client):
    r = client.get("/experiment/config")
    assert r.status_code == 200
    data = r.json()
    assert "rideshare" in data["app_types"]
    assert "privacy_preserving" in data["reason_types"]


def test_assign_then_reassign_is_stable(client):
    pid = str(uuid.uuid4())
    r1 = client.post("/experiment/assign", json={"participant_id": pid})
    r2 = client.post("/experiment/assign", json={"participant_id": pid})
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["condition"] == r2.json()["condition"]


def test_event_requires_known_participant(client):
    r = client.post(
        "/events/permission",
        json={"participant_id": "does-not-exist", "event": "permission_result"},
    )
    assert r.status_code == 404


def test_full_flow(client):
    pid = str(uuid.uuid4())
    assign_resp = client.post("/experiment/assign", json={"participant_id": pid})
    assert assign_resp.status_code == 200

    event_resp = client.post(
        "/events/permission",
        json={
            "participant_id": pid,
            "event": "permission_result",
            "decision": "granted",
            "precision": "approximate",
            "response_time_ms": 1500,
        },
    )
    assert event_resp.status_code == 200

    survey_resp = client.post(
        "/surveys/submit",
        json={
            "participant_id": pid,
            "app_trust": 4,
            "android_trust": 3,
            "necessity": 4,
            "privacy_concern": 2,
        },
    )
    assert survey_resp.status_code == 200

    events = client.get(f"/events/permission/{pid}")
    assert events.status_code == 200
    assert len(events.json()) == 1
