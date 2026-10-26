from __future__ import annotations
import argparse
import pandas as pd
from embedbench.optimizer import choose_configuration

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="outputs/results.csv")
    p.add_argument("--min-ndcg", type=float, default=0.5)
    p.add_argument("--max-storage-bytes", type=int)
    p.add_argument("--max-query-ms", type=float)
    args = p.parse_args()
    df = pd.read_csv(args.results)
    result = choose_configuration(df, args.min_ndcg, args.max_storage_bytes, args.max_query_ms)
    if result is None:
        print("No tested configuration satisfies all constraints.")
    else:
        print("Selected configuration:")
        for key, value in result.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
