import json
import sqlite3
from pathlib import Path
DB_PATH = Path("flowfix.db")

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS repair_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL,
            status TEXT NOT NULL, elapsed_ms REAL NOT NULL, attempts INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP, payload TEXT NOT NULL)""")

def save_result(result):
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT INTO repair_runs(case_id,status,elapsed_ms,attempts,payload) VALUES(?,?,?,?,?)",
                    (result.get("case_id",""), result.get("status",""), result.get("elapsed_ms",0),
                     result.get("attempts",0), json.dumps(result)))

def recent_runs(limit=100):
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute("SELECT id,case_id,status,elapsed_ms,attempts,created_at,payload FROM repair_runs ORDER BY id DESC LIMIT ?",
                           (limit,)).fetchall()
    return [{"id":r[0],"case_id":r[1],"status":r[2],"elapsed_ms":r[3],"attempts":r[4],
             "created_at":r[5],"payload":json.loads(r[6])} for r in rows]
