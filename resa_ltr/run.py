import argparse
from pathlib import Path
from src.experiments import run_once, validation_debias_experiment, save_report

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=700)
    p.add_argument("--impressions", type=int, default=24)
    p.add_argument("--candidates", type=int, default=20)
    p.add_argument("--bias", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("="*68)
    print("RESA-LTR | POSITION-BIAS DEBIASING")
    print("="*68)
    result = run_once(
        args.users, args.impressions, args.candidates, args.bias, args.seed
    )
    validation = validation_debias_experiment(min(args.users, 300), args.seed+1)
    save_report(result, validation)

    print("\nMODEL RESULTS")
    for method, metrics in result["metrics"].items():
        print(f"\n{method}")
        for metric, value in metrics.items():
            print(f"  {metric:22s}: {value:.4f}")

    print("\nSYSTEM")
    print(f"  train rows      : {result['rows_train']:,}")
    print(f"  test rows       : {result['rows_test']:,}")
    print(f"  elapsed seconds : {result['seconds']:.2f}")

    print("\nARTIFACTS")
    for f in sorted(Path("artifacts").glob("*")):
        print(" ", f)

if __name__ == "__main__":
    main()
