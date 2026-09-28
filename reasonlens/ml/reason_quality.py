"""
Reason-quality ML model.

Pipeline: reason_text -> TF-IDF embedding -> quality model.
Two supervised models share the same TF-IDF features:
  1. A classifier predicting reason_type from text (sanity-check task).
  2. A regressor predicting the continuous quality_score (the model used
     downstream to test whether reason quality predicts permission
     decisions and trust).
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline


def build_classifier_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def build_rf_classifier_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )


def build_quality_regressor_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
            ("reg", Ridge(alpha=1.0)),
        ]
    )


def build_rf_quality_regressor_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
            ("reg", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )
