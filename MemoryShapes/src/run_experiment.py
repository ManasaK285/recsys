import argparse
from .evaluate import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run MemoryShapes experiment.")
    parser.add_argument("--participants", type=int, default=120)
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics, _ = run_experiment(args.participants, args.trials, args.epochs, args.seed)
    print("\nHeld-out participant metrics (lower log loss is better):")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
