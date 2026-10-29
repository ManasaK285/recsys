from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "signalscope.duckdb"
PARQUET_PATH = DATA_DIR / "telemetry.parquet"

class TelemetryStore:
    def __init__(self, db_path=DB_PATH, parquet_path=PARQUET_PATH):
        self.db_path, self.parquet_path = Path(db_path), Path(parquet_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.db_path)) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS telemetry (
              timestamp TIMESTAMPTZ, service VARCHAR, region VARCHAR, release VARCHAR,
              latency_p50_ms DOUBLE, latency_p95_ms DOUBLE, error_rate DOUBLE,
              cpu_pct DOUBLE, memory_pct DOUBLE, request_count BIGINT, queue_depth BIGINT,
              is_injected_anomaly BOOLEAN)""")

    def count(self):
        with duckdb.connect(str(self.db_path), read_only=True) as con:
            return int(con.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0])

    def replace(self, df):
        with duckdb.connect(str(self.db_path)) as con:
            con.register("incoming", df)
            con.execute("DELETE FROM telemetry")
            con.execute("INSERT INTO telemetry SELECT * FROM incoming")
            con.unregister("incoming")
            path = self.parquet_path.as_posix().replace("'", "''")
            con.execute(f"COPY (SELECT * FROM telemetry ORDER BY timestamp) TO '{path}' (FORMAT PARQUET)")

    def append(self, df):
        if df.empty:
            return self.count()
        with duckdb.connect(str(self.db_path)) as con:
            con.register("incoming", df)
            con.execute("INSERT INTO telemetry SELECT * FROM incoming")
            con.unregister("incoming")
            path = self.parquet_path.as_posix().replace("'", "''")
            con.execute(f"COPY (SELECT * FROM telemetry ORDER BY timestamp) TO '{path}' (FORMAT PARQUET)")
            return int(con.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0])

    def read(self):
        with duckdb.connect(str(self.db_path), read_only=True) as con:
            return con.execute("SELECT * FROM telemetry ORDER BY timestamp").df()
