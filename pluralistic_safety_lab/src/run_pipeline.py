"""CLI entry point: python -m src.run_pipeline."""
from __future__ import annotations
import argparse
from .analysis import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pluralistic ordinal safety ratings.")
    parser.add_argument("--input", default="data/ratings.csv", help="Input CSV. Missing file uses synthetic demo data.")
    parser.add_argument("--output-dir", default="reports", help="Where to write outputs.")
    parser.add_argument("--no-synthetic", action="store_true", help="Fail if input CSV does not exist.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = run_analysis(args.input, args.output_dir, args.no_synthetic, args.seed)
    print("Analysis complete.")
    print(f"Source: {summary['data_source']}")
    print(f"Ratings: {summary['n_ratings']:,} | Items: {summary['n_items']:,} | "
          f"Raters: {summary['n_raters']:,} | Groups: {summary['n_groups']}")
    print(f"Outputs written to: {args.output_dir}")
    print("Top group-level summaries:")
    for row in summary["group_metrics"]:
        print(f"  {row['group']}: span={row['scale_span']:.3f}, "
              f"monotonicity={row['monotonicity']:.3f}, "
              f"Spearman={row['severity_spearman']:.3f}, "
              f"entropy={row['rating_entropy']:.3f}")


if __name__ == "__main__":
    main()
