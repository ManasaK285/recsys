import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.concepts import load_concepts
from src.experiments.novel_concepts import build_grounded_descriptions


def main():
    concepts = load_concepts(ROOT / "data/concepts/novel.json")
    rows = build_grounded_descriptions(concepts)

    print("\nCONTROLLED NOVEL-CONCEPT EXPERIENCES\n")

    for row in rows:
        print(f"{row['concept']}:")
        print(f"  {row['description']}")
        print()


if __name__ == "__main__":
    main()
