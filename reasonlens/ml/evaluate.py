"""Computes evaluation metrics for the reason-quality classifier and regressor."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import label_binarize


def evaluate_classifier(clf, X_test, y_test, classes) -> dict:
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    roc_auc = None
    try:
        y_proba = clf.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=classes)
        roc_auc = float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"))
    except Exception:
        pass

    return {
        "task": "reason_type_classification",
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "roc_auc_macro": roc_auc,
    }


def evaluate_regressor(reg, X_test, y_test) -> dict:
    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    # simple calibration diagnostic: correlation between predicted and actual deciles
    order = np.argsort(y_pred)
    calibration_corr = float(np.corrcoef(np.array(y_pred)[order], np.array(y_test)[order])[0, 1])

    return {
        "task": "quality_score_regression",
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "calibration_correlation": calibration_corr,
    }


def run_evaluation(clf, reg, split: dict) -> dict:
    classes = sorted(set(split["yc_test"]))
    clf_metrics = evaluate_classifier(clf, split["X_test"], split["yc_test"], classes)
    reg_metrics = evaluate_regressor(reg, split["X_test"], split["yr_test"])
    return {"classifier": clf_metrics, "regressor": reg_metrics}
