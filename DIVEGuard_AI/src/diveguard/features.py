from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from .config import DEMOGRAPHIC_COLUMNS

def available_demographics(df: pd.DataFrame) -> list[str]:
    return [c for c in DEMOGRAPHIC_COLUMNS if c in df.columns]

def prepare_features(df: pd.DataFrame, pluralistic: bool = False) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["text"] = (
        df["text_prompt"].fillna("").astype(str) + " [TOPIC] " +
        df.get("topic", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    )
    if pluralistic:
        demo_cols = available_demographics(df)
        for col in demo_cols:
            x[col] = df[col].fillna("unknown").astype(str)
        if demo_cols:
            # Interaction feature allows the model to learn context-specific variation.
            x["context_topic"] = (
                df[demo_cols].fillna("unknown").astype(str).agg("|".join, axis=1)
                + " :: " + df.get("topic", pd.Series("unknown", index=df.index)).astype(str)
            )
    return x

def build_model(pluralistic=False, categorical_columns=None):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    transformers = [
        (
            "text",
            TfidfVectorizer(
                max_features=30000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True
            ),
            "text"
        )
    ]

    categorical_columns = categorical_columns or []

    if pluralistic and categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2
                ),
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1200,
                class_weight="balanced",
                C=2.0,
                solver="lbfgs"
            )
        )
    ])

    return model