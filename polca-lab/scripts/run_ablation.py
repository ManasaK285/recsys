import json, subprocess, sys
from pathlib import Path

BASE = json.loads(Path("configs/default.json").read_text())
EXPS = {
    "full": {},
    "no_epsilon": {"use_epsilon_net": False, "output_prefix": "ablation_no_epsilon"},
    "no_summary": {"use_summary": False, "output_prefix": "ablation_no_summary"},
    "ucb": {"priority": "ucb", "output_prefix": "ablation_ucb"},
}
for name, patch in EXPS.items():
    cfg = dict(BASE); cfg.update(patch)
    if name == "full": cfg["output_prefix"] = "ablation_full"
    tmp = Path("configs") / f"_tmp_{name}.json"
    tmp.write_text(json.dumps(cfg, indent=2))
    subprocess.run([sys.executable, "scripts/run_experiment.py", "--config", str(tmp)], check=True)
    tmp.unlink()
