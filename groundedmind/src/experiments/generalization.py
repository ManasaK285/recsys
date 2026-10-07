import itertools


def held_out_combinations(values_a, values_b):
    return list(itertools.product(values_a, values_b))


def composition_accuracy(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have equal length.")
    if not targets:
        return 0.0
    return sum(p == t for p, t in zip(predictions, targets)) / len(targets)
