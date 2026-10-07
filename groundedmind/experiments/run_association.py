import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.concepts import load_concepts
from src.models.text_model import TextModel
from src.experiments.association import run_association_experiment


def main():
    concepts = (
        load_concepts(ROOT / "data/concepts/concrete.json")
        + load_concepts(ROOT / "data/concepts/abstract.json")
    )

    print("Loading sentence-transformer...")
    model = TextModel()

    rows = run_association_experiment(model, concepts)

    output = ROOT / "results" / "association_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(output, index=False)

    print(f"Saved: {output}")

    df = pd.DataFrame(rows)
    for concept in concepts[:5]:
        subset = df[df["concept"] == concept].sort_values(
            "probability", ascending=False
        )
        print(f"\n{concept}:")
        print(subset.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
