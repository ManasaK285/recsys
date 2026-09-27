import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]


def association_distribution(model, concept, candidates=COLORS):
    concept_embedding = model.encode(concept)
    candidate_embeddings = np.asarray(model.encode_many(candidates))

    similarities = cosine_similarity(
        concept_embedding.reshape(1, -1),
        candidate_embeddings,
    )[0]

    scores = np.exp(similarities - similarities.max())
    probabilities = scores / scores.sum()

    return dict(zip(candidates, probabilities.astype(float)))


def run_association_experiment(model, concepts, candidates=COLORS):
    rows = []
    for concept in concepts:
        distribution = association_distribution(model, concept, candidates)
        for candidate, probability in distribution.items():
            rows.append({
                "concept": concept,
                "candidate": candidate,
                "probability": probability,
            })
    return rows
