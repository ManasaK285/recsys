import argparse
from pathlib import Path
from preferenceforge.data import generate_dataset, save_dataset
from preferenceforge.train import train_all
from preferenceforge.evaluate import evaluate

def main():
    p = argparse.ArgumentParser(description="PreferenceForge research pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--users", type=int, default=30); g.add_argument("--pairs-per-user", type=int, default=80); g.add_argument("--seed", type=int, default=42); g.add_argument("--heterogeneity", type=float, default=0.8)
    sub.add_parser("train"); sub.add_parser("evaluate"); sub.add_parser("all")
    args = p.parse_args()
    if args.cmd == "generate" or args.cmd == "all":
        df, weights = generate_dataset(getattr(args, "users", 30), getattr(args, "pairs_per_user", 80), getattr(args, "seed", 42), getattr(args, "heterogeneity", 0.8))
        save_dataset(df)
        Path("artifacts").mkdir(exist_ok=True)
        import json
        Path("artifacts/simulated_user_weights.json").write_text(json.dumps(weights, indent=2))
        print(f"Generated {len(df)} pairwise comparisons.")
    if args.cmd in ("train", "all"):
        models, train, test = train_all()
        print(f"Trained models on {len(train)} rows; held out {len(test)} rows.")
    if args.cmd in ("evaluate", "all"):
        print(evaluate().to_string(index=False))

if __name__ == "__main__":
    main()
