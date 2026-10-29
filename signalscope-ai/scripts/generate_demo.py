import argparse
from signalscope.simulator import generate_telemetry
from signalscope.storage import TelemetryStore

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--services", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--interval-minutes", type=int, choices=[5,15,30,60], default=60)
    args = p.parse_args()
    df = generate_telemetry(args.days, args.services, seed=args.seed, interval_minutes=args.interval_minutes)
    TelemetryStore().replace(df)
    print(f"Generated {len(df):,} rows; {df.service.nunique()} services; {df.region.nunique()} regions.")
    print(f"Injected anomaly rows: {int(df.is_injected_anomaly.sum()):,}")
    print("Saved data/signalscope.duckdb and data/telemetry.parquet")

if __name__ == "__main__":
    main()
