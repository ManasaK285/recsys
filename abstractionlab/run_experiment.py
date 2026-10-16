import argparse
import json
from abstractionlab.experiments import run_all

def main():
    parser = argparse.ArgumentParser(description="Run AbstractionLab toy experiments.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=float, default=0.025)
    parser.add_argument("--target", type=float, default=0.75)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    result = run_all(args.steps, args.seed, args.noise, args.target, args.out)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
