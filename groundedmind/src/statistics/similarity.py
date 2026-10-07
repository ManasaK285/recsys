import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon


def _vector(distribution, categories):
    return np.asarray(
        [distribution.get(category, 0.0) for category in categories],
        dtype=float,
    )


def distribution_correlation(human, model, categories):
    h = _vector(human, categories)
    m = _vector(model, categories)

    pearson = pearsonr(h, m)
    spearman = spearmanr(h, m)

    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def js_distance(distribution_a, distribution_b, categories):
    a = _vector(distribution_a, categories)
    b = _vector(distribution_b, categories)

    # Normalize defensively.
    a = a / a.sum() if a.sum() else np.ones(len(a)) / len(a)
    b = b / b.sum() if b.sum() else np.ones(len(b)) / len(b)

    return float(jensenshannon(a, b))
