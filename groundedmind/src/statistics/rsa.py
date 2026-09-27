import numpy as np
from scipy.stats import spearmanr


def rsa_similarity(matrix_a, matrix_b):
    matrix_a = np.asarray(matrix_a, dtype=float)
    matrix_b = np.asarray(matrix_b, dtype=float)

    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Matrices must have the same shape.")

    idx = np.triu_indices_from(matrix_a, k=1)
    a = matrix_a[idx]
    b = matrix_b[idx]

    result = spearmanr(a, b)

    return {
        "spearman_r": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def pairwise_cosine_matrix(embeddings):
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(np.asarray(embeddings))
