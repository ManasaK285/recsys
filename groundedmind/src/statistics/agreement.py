import pandas as pd


def response_distribution(dataframe, concept, column):
    subset = dataframe[dataframe["concept"] == concept]
    if subset.empty:
        return {}
    return subset[column].value_counts(normalize=True).to_dict()


def majority_response(dataframe, concept, column):
    distribution = response_distribution(dataframe, concept, column)
    if not distribution:
        return None
    return max(distribution, key=distribution.get)
