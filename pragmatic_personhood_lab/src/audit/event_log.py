import hashlib
import json
from datetime import datetime, timezone
from src.database import connect

GENESIS = "0" * 64

def _canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

def append_event(event_type: str, agent_id: str | None, payload: dict):
    timestamp = datetime.now(timezone.utc).isoformat()
    with connect() as con:
        previous = con.execute("SELECT event_hash FROM audit_events ORDER BY id DESC LIMIT 1").fetchone()
        prev_hash = previous["event_hash"] if previous else GENESIS
        body = {"timestamp": timestamp, "event_type": event_type, "agent_id": agent_id,
                "payload": payload, "previous_hash": prev_hash}
        event_hash = hashlib.sha256(_canonical(body).encode()).hexdigest()
        con.execute("""INSERT INTO audit_events(timestamp,event_type,agent_id,payload_json,previous_hash,event_hash)
                       VALUES(?,?,?,?,?,?)""",
                    (timestamp, event_type, agent_id, json.dumps(payload, sort_keys=True),
                     prev_hash, event_hash))
    return {"timestamp": timestamp, "event_hash": event_hash, "previous_hash": prev_hash}

def list_events(limit=500):
    with connect() as con:
        rows = con.execute("SELECT * FROM audit_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) | {"payload": json.loads(r["payload_json"])} for r in rows]

def verify_chain():
    with connect() as con:
        rows = con.execute("SELECT * FROM audit_events ORDER BY id ASC").fetchall()
    previous = GENESIS
    for row in rows:
        body = {"timestamp": row["timestamp"], "event_type": row["event_type"],
                "agent_id": row["agent_id"], "payload": json.loads(row["payload_json"]),
                "previous_hash": previous}
        expected = hashlib.sha256(_canonical(body).encode()).hexdigest()
        if row["previous_hash"] != previous or row["event_hash"] != expected:
            return {"valid": False, "failed_event_id": row["id"], "events_checked": len(rows)}
        previous = row["event_hash"]
    return {"valid": True, "failed_event_id": None, "events_checked": len(rows)}
