import numpy as np
import pandas as pd
from src.models import q_learning_predict, episodic_memory_predict, sequence_tensor_for_group


def sample_df():
    return pd.DataFrame({
        "participant_id": [1, 1, 1],
        "trial": [0, 1, 2],
        "choice": [0, 1, 0],
        "reward": [1, 0, 1],
    })


def test_baseline_probabilities_sum_to_one():
    df = sample_df()
    for fn in [q_learning_predict, episodic_memory_predict]:
        out = fn(df)
        assert len(out) == 3
        for row in out:
            assert np.isclose(row[3].sum(), 1.0)


def test_sequence_is_causal():
    X, y = sequence_tensor_for_group(sample_df())
    assert X.shape == (3, 3, 5)
    assert np.all(X[0] == 0)
    assert np.all(X[1, 0, :].sum() > 0)
    assert X[1, 1, :].sum() == 0
