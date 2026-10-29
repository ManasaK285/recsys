import numpy as np
import pandas as pd

METRICS = ["latency_p95_ms", "error_rate", "cpu_pct", "memory_pct", "queue_depth"]

def detect_anomalies(df):
    if df.empty:
        out = df.copy()
        out["is_detected_anomaly"] = pd.Series(dtype=bool)
        out["anomaly_score"] = pd.Series(dtype=float)
        out["severity"] = pd.Series(dtype=str)
        out["triggered_metrics"] = pd.Series(dtype=str)
        return out
    work = df.copy()
    older_release = sorted(work["release"].dropna().unique())[:1]
    baseline = work[work.release == older_release[0]] if older_release else work
    stats = baseline.groupby(["service", "region"])[METRICS].agg(["median", lambda x: x.quantile(.75) - x.quantile(.25)]).reset_index()
    names = ["service", "region"]
    for metric in METRICS:
        names += [f"{metric}_median", f"{metric}_iqr"]
    stats.columns = names
    work = work.merge(stats, on=["service", "region"], how="left")
    masks, scores = [], []
    for metric in METRICS:
        med = work[f"{metric}_median"].fillna(work[metric].median())
        iqr = work[f"{metric}_iqr"].fillna(0).clip(lower=0)
        val = work[metric]
        if metric == "latency_p95_ms":
            threshold = np.maximum(med * 1.25, med + 3 * iqr)
        elif metric == "error_rate":
            threshold = med + np.maximum(.008, 3 * iqr)
        elif metric in ("cpu_pct", "memory_pct"):
            threshold = med + np.maximum(15, 3 * iqr)
        else:
            threshold = med + np.maximum(10, 3 * iqr)
        masks.append((val > threshold).to_numpy())
        scores.append(((val - med) / np.maximum(np.abs(med), 1)).clip(lower=0).to_numpy())
    matrix = np.column_stack(masks)
    work["is_detected_anomaly"] = matrix.any(axis=1)
    work["anomaly_score"] = np.max(np.column_stack(scores), axis=1)
    work["severity"] = np.select([work.anomaly_score >= .75, work.anomaly_score >= .35], ["high", "medium"], default="low")
    metric_names = np.array(METRICS, dtype=object)
    work["triggered_metrics"] = [", ".join(metric_names[row].tolist()) for row in matrix]
    return work.drop(columns=[c for c in work.columns if c.endswith("_median") or c.endswith("_iqr")])

def summarize(df):
    if df.empty:
        return {"row_count": 0, "service_count": 0, "region_count": 0, "release_count": 0}
    return {
        "row_count": int(len(df)), "service_count": int(df.service.nunique()),
        "region_count": int(df.region.nunique()), "release_count": int(df.release.nunique()),
        "time_start": pd.Timestamp(df.timestamp.min()).isoformat(),
        "time_end": pd.Timestamp(df.timestamp.max()).isoformat(),
        "mean_latency_p95_ms": round(float(df.latency_p95_ms.mean()), 3),
        "mean_error_rate": round(float(df.error_rate.mean()), 6),
        "total_requests": int(df.request_count.sum())
    }

def release_comparison(df, service=None, region=None):
    work = df.copy()
    if service: work = work[work.service == service]
    if region: work = work[work.region == region]
    releases = sorted(work.release.dropna().unique().tolist())
    if len(releases) < 2:
        return {"available": False, "message": "At least two releases are required.", "releases": releases, "metrics": []}
    old, new = releases[-2], releases[-1]
    metrics, rows = ["latency_p95_ms", "error_rate", "cpu_pct", "memory_pct", "queue_depth", "request_count"], []
    for metric in metrics:
        a = work.loc[work.release == old, metric].astype(float)
        b = work.loc[work.release == new, metric].astype(float)
        if a.empty or b.empty: continue
        av, bv = float(a.mean()), float(b.mean())
        delta = bv - av
        pct = delta / abs(av) * 100 if av else None
        rows.append({"metric": metric, "older_release": old, "newer_release": new,
          "older_mean": round(av, 5), "newer_mean": round(bv, 5),
          "absolute_change": round(delta, 5), "percent_change": round(pct, 3) if pct is not None else None,
          "older_sample_count": int(len(a)), "newer_sample_count": int(len(b))})
    return {"available": True, "older_release": old, "newer_release": new, "service_filter": service, "region_filter": region, "metrics": rows}

def evaluation_metrics(df):
    detected = detect_anomalies(df)
    actual = detected.is_injected_anomaly.astype(bool).to_numpy()
    pred = detected.is_detected_anomaly.astype(bool).to_numpy()
    tp, fp, tn, fn = int((actual & pred).sum()), int((~actual & pred).sum()), int((~actual & ~pred).sum()), int((actual & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"rows": len(df), "ground_truth_anomalies": int(actual.sum()), "detected_anomalies": int(pred.sum()),
      "true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn,
      "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
      "note": "Synthetic-data metrics; not a production accuracy estimate."}
