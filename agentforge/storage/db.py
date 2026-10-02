from __future__ import annotations
import json, os, sqlite3
from pathlib import Path

class Database:
    def __init__(self, path: str|None=None):
        self.path=path or os.getenv('AGENTFORGE_DB_PATH','agentforge.db')
        self.init()
    def conn(self): return sqlite3.connect(self.path)
    def init(self):
        c=self.conn(); c.executescript('''
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, task TEXT, status TEXT, attempts INTEGER, result_json TEXT);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, event TEXT, data_json TEXT);
        CREATE TABLE IF NOT EXISTS harness_versions(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, version TEXT, intervention_json TEXT);
        '''); c.commit(); c.close()
    def save_run(self, run_id, task, status, attempts, result):
        c=self.conn(); c.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)',(run_id,task,status,attempts,json.dumps(result))); c.commit(); c.close()
    def save_event(self, run_id,event,data):
        c=self.conn(); c.execute('INSERT INTO events(run_id,event,data_json) VALUES (?,?,?)',(run_id,event,json.dumps(data,default=str))); c.commit(); c.close()
    def get_run(self, run_id):
        c=self.conn(); row=c.execute('SELECT id,task,status,attempts,result_json FROM runs WHERE id=?',(run_id,)).fetchone(); c.close()
        if not row:return None
        return {'id':row[0],'task':row[1],'status':row[2],'attempts':row[3],'result':json.loads(row[4])}
