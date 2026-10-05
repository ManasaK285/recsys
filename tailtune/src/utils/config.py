from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config(path=None):
    path = Path(path or ROOT / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def project_path(relative):
    return ROOT / relative
