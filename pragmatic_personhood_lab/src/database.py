import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "personhood_lab.db"

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS agents (
          agent_id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL,
          owner TEXT NOT NULL, permissions_json TEXT NOT NULL,
          purchase_limit REAL NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'active',
          require_human_approval_for_external INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS decisions (
          id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
          agent_id TEXT NOT NULL, action_json TEXT NOT NULL, decision TEXT NOT NULL,
          reason TEXT NOT NULL, policy_version TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
          event_type TEXT NOT NULL, agent_id TEXT, payload_json TEXT NOT NULL,
          previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL
        );
        """)

def upsert_agent(agent: dict[str, Any]):
    with connect() as con:
        con.execute("""
        INSERT INTO agents(agent_id,name,role,owner,permissions_json,purchase_limit,status,require_human_approval_for_external)
        VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(agent_id) DO UPDATE SET name=excluded.name, role=excluded.role,
          owner=excluded.owner, permissions_json=excluded.permissions_json,
          purchase_limit=excluded.purchase_limit, status=excluded.status,
          require_human_approval_for_external=excluded.require_human_approval_for_external
        """, (agent["agent_id"], agent["name"], agent["role"], agent["owner"],
              json.dumps(agent["permissions"]), float(agent.get("purchase_limit", 0)),
              agent.get("status", "active"), int(agent.get("require_human_approval_for_external", True))))

def get_agent(agent_id: str):
    with connect() as con:
        row = con.execute("SELECT * FROM agents WHERE agent_id=?", (agent_id,)).fetchone()
    return row_to_agent(row) if row else None

def list_agents():
    with connect() as con:
        rows = con.execute("SELECT * FROM agents ORDER BY name").fetchall()
    return [row_to_agent(r) for r in rows]

def row_to_agent(row):
    return {
        "agent_id": row["agent_id"], "name": row["name"], "role": row["role"],
        "owner": row["owner"], "permissions": json.loads(row["permissions_json"]),
        "purchase_limit": row["purchase_limit"], "status": row["status"],
        "require_human_approval_for_external": bool(row["require_human_approval_for_external"])
    }

def save_decision(timestamp, action, decision, reason, policy_version="1.0"):
    with connect() as con:
        con.execute("""INSERT INTO decisions(timestamp,agent_id,action_json,decision,reason,policy_version)
                       VALUES(?,?,?,?,?,?)""",
                    (timestamp, action["agent_id"], json.dumps(action), decision, reason, policy_version))

def list_decisions(limit=500):
    with connect() as con:
        rows = con.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) | {"action": json.loads(r["action_json"])} for r in rows]

def seed_agents():
    if list_agents():
        return
    examples = [
      {"agent_id":"agent-procurement-001","name":"Procurement Agent","role":"procurement","owner":"Operations Team",
       "permissions":["read_suppliers","compare_prices","create_purchase_order","approve_purchase"],
       "purchase_limit":5000,"status":"active","require_human_approval_for_external":True},
      {"agent_id":"agent-research-001","name":"Research Agent","role":"research","owner":"Research Lead",
       "permissions":["read_approved_data","summarize_data","create_report"],
       "purchase_limit":0,"status":"active","require_human_approval_for_external":True},
      {"agent_id":"agent-support-001","name":"Customer Support Agent","role":"support","owner":"Support Manager",
       "permissions":["read_customer_ticket","draft_response","issue_refund"],
       "purchase_limit":100,"status":"active","require_human_approval_for_external":True},
    ]
    for agent in examples:
        upsert_agent(agent)
