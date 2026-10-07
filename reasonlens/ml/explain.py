"""
Explainability for the reason-quality models.

Primary method: inspect TF-IDF feature coefficients directly (fast, always
available, and easy to sanity-check by eye). Optionally augmented with
SHAP values on a small sample when the `shap` package is installed, since
KernelExplainer on a full TF-IDF vocabulary is expensive.
"""
import numpy as np


def top_terms_for_regressor(reg_pipeline, top_k: int = 15) -> dict:
    vectorizer = reg_pipeline.named_steps["tfidf"]
    model = reg_pipeline.named_steps["reg"]
    feature_names = np.array(vectorizer.get_feature_names_out())

    if not hasattr(model, "coef_"):
        return {"note": f"{type(model).__name__} has no linear coefficients to inspect"}

    coefs = model.coef_
    order = np.argsort(coefs)
    top_negative = [
        {"term": feature_names[i], "coef": float(coefs[i])} for i in order[:top_k]
    ]
    top_positive = [
        {"term": feature_names[i], "coef": float(coefs[i])} for i in order[-top_k:][::-1]
    ]
    return {
        "task": "quality_score_regression",
        "top_terms_increasing_quality": top_positive,
        "top_terms_decreasing_quality": top_negative,
    }


def top_terms_for_classifier(clf_pipeline, top_k: int = 10) -> dict:
    vectorizer = clf_pipeline.named_steps["tfidf"]
    model = clf_pipeline.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())

    if not hasattr(model, "coef_"):
        return {"note": f"{type(model).__name__} has no linear coefficients to inspect"}

    result = {}
    for class_idx, class_label in enumerate(model.classes_):
        coefs = model.coef_[class_idx]
        order = np.argsort(coefs)[-top_k:][::-1]
        result[str(class_label)] = [
            {"term": feature_names[i], "coef": float(coefs[i])} for i in order
        ]
    return {"task": "reason_type_classification", "top_terms_by_class": result}


def shap_summary(reg_pipeline, sample_texts, max_samples: int = 40) -> dict:
    """Best-effort SHAP explanation on a small sample; returns a note if
    SHAP is unavailable or the model type isn't supported, rather than
    failing the whole pipeline."""
    try:
        import shap
    except ImportError:
        return {"note": "shap not installed; skipping SHAP explanation"}

    try:
        sample = list(sample_texts)[:max_samples]
        vectorizer = reg_pipeline.named_steps["tfidf"]
        model = reg_pipeline.named_steps["reg"]
        X_sample = vectorizer.transform(sample).toarray()

        if hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X_sample)
        else:
            explainer = shap.Explainer(model.predict, X_sample)

        shap_values = explainer(X_sample)
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        feature_names = np.array(vectorizer.get_feature_names_out())
        order = np.argsort(mean_abs)[-15:][::-1]
        return {
            "note": f"SHAP computed on {len(sample)} sampled texts",
            "top_features_by_mean_abs_shap": [
                {"term": feature_names[i], "mean_abs_shap": float(mean_abs[i])} for i in order
            ],
        }
    except Exception as e:
        return {"note": f"SHAP explanation failed: {e}"}
