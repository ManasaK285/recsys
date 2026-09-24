import json
from pathlib import Path


def load_concepts(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_all_concepts(base_dir="data/concepts"):
    base = Path(base_dir)
    return {
        "concrete": load_concepts(base / "concrete.json"),
        "abstract": load_concepts(base / "abstract.json"),
        "novel": load_concepts(base / "novel.json"),
    }
