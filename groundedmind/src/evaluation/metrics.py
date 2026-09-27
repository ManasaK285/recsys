import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


def classification_accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def brier_score(probabilities, outcomes):
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((probabilities - outcomes) ** 2))
