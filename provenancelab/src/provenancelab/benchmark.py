"""Reproducible pipeline smoke benchmark; not a validated detector benchmark."""
import random
from .detectors import DemoBaselineDetector
from .transformations import transform
from .metrics import binary_metrics

HUMAN_LIKE = [
    "I tried the new route this morning, and it took about twenty minutes longer than expected.",
    "The meeting moved to Thursday because two people could not make the original time.",
    "We tested three settings and kept the second one; it was easier to maintain.",
    "My notes are a little messy, but the main issue is the missing data from Tuesday.",
    "The library was quiet after lunch, so I finished the draft before heading home.",
]
GENERATED_LIKE = [
    "In conclusion, it is important to note that the proposed approach provides a comprehensive overview of the topic.",
    "Furthermore, this solution demonstrates a structured and effective way to address the challenge.",
    "Moreover, the results highlight the importance of a consistent and scalable methodology.",
    "It is important to note that this framework can support a wide range of potential applications.",
    "In conclusion, the analysis provides valuable insights and a comprehensive understanding of the problem.",
]

def run_benchmark(samples_per_class=30, seed=42):
    rng = random.Random(seed)
    detector = DemoBaselineDetector()
    rows = []
    for label, pool in [(0, HUMAN_LIKE), (1, GENERATED_LIKE)]:
        for i in range(samples_per_class):
            text = pool[i % len(pool)]
            # Controlled deterministic variation to exercise the pipeline.
            if i % 3 == 1:
                text = transform(text, "truncate")
            elif i % 3 == 2:
                text = transform(text, "light_edit")
            score = detector.score(text)
            pred = int(score >= 0.48)
            rows.append({"label": label, "prediction": pred, "score": score, "transformation": ["original", "truncate", "light_edit"][i % 3]})
    rng.shuffle(rows)
    metrics = binary_metrics([r["label"] for r in rows], [r["prediction"] for r in rows])
    return {"metrics": metrics, "rows": rows, "seed": seed, "samples_per_class": samples_per_class,
            "warning": "Synthetic smoke test using a toy lexical baseline; results are not evidence of real-world AI-authorship detection."}
