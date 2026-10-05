import subprocess
import sys
from pathlib import Path
import yaml

def main():
    # Train each alpha. Evaluation is intentionally left as a separate command
    # so experiments can be run independently on CPU/GPU.
    alphas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    for alpha in alphas:
        cmd = [sys.executable, "-m", "src.experiments.train_bt_sr",
               "--alpha", str(alpha)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
