import argparse
from pathlib import Path
from collectivealign.experiment import load_config, run_experiment
from collectivealign.reporting import make_reports

def main():
    parser = argparse.ArgumentParser(description="Run CollectiveAlign Lab experiment")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    config = load_config(args.config)
    df = run_experiment(config, args.out)
    make_reports(df, args.out)
    print(f"Completed: {len(df)} condition/model/protocol rows")
    print(f"Outputs written to: {Path(args.out).resolve()}")
    print("All generated observations are synthetic.")
if __name__ == "__main__":
    main()
