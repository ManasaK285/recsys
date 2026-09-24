import json
import sqlite3
from pathlib import Path
from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    parent_id TEXT,
    depth INTEGER NOT NULL,
    approach TEXT NOT NULL,
    code TEXT NOT NULL,
    evaluation_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    policy_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policy_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id TEXT NOT NULL,
    world_id TEXT NOT NULL,
    reward REAL NOT NULL,
    work INTEGER NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

def connect():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with connect() as c:
        c.executescript(SCHEMA)
        c.commit()

def insert_task(task_id, name, description):
    with connect() as c:
        c.execute("INSERT OR IGNORE INTO tasks VALUES (?, ?, ?)", (task_id, name, description))
        c.commit()

def insert_run(run_id, task_id, policy_version, cycle, created_at):
    with connect() as c:
        c.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
                  (run_id, task_id, policy_version, cycle, created_at))
        c.commit()

def insert_node(node):
    with connect() as c:
        c.execute(
            """INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (node.node_id, node.run_id, node.task_id, node.parent_id, node.depth,
             node.approach, node.code, json.dumps(node.evaluation.to_dict()),
             node.created_at, node.policy_version)
        )
        c.commit()

def get_run(run_id):
    with connect() as c:
        run = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        nodes = c.execute("SELECT * FROM nodes WHERE run_id=? ORDER BY depth, created_at",
                           (run_id,)).fetchall()
    return run, nodes

def get_task(task_id):
    with connect() as c:
        return c.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()

def get_nodes(task_id):
    with connect() as c:
        return c.execute("SELECT * FROM nodes WHERE task_id=? ORDER BY created_at",
                         (task_id,)).fetchall()

def insert_policy(policy_id, name, source, created_at):
    with connect() as c:
        c.execute("INSERT OR REPLACE INTO policies VALUES (?, ?, ?, ?)",
                  (policy_id, name, source, created_at))
        c.commit()

def insert_policy_eval(policy_id, world_id, reward, work, details, created_at):
    with connect() as c:
        c.execute(
            "INSERT INTO policy_evaluations(policy_id,world_id,reward,work,details_json,created_at) VALUES (?,?,?,?,?,?)",
            (policy_id, world_id, reward, work, json.dumps(details), created_at)
        )
        c.commit()

def get_policy_evals():
    with connect() as c:
        return c.execute("SELECT * FROM policy_evaluations ORDER BY id DESC").fetchall()
