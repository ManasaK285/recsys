"""Concern extraction (lexicon) + semantic clustering (TF-IDF or sentence embeddings) + validity checks."""
import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score

LEXICON = {
    "accessibility": r"accessib|disabilit|screen reader|text-to-speech|captioning|assistive",
    "privacy": r"privacy|student data|data collection|personal information|surveillance|tracking",
    "academic_integrity": r"cheat|integrity|plagiar|honest|original work|detector",
    "language_access": r"multilingual|english learner|translation|translate|native language|language barrier|language access",
    "teacher_workload": r"workload|grading time|extra work|teacher time|training time|burnout|overwhelm",
}
CONCERNS = list(LEXICON)


def extract_concerns(text: str) -> list:
    t = text.lower()
    return [c for c, pat in LEXICON.items() if re.search(pat, t)]


def add_concerns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["concerns"] = df["text"].apply(extract_concerns)
    return df


def evaluate_extractor(df: pd.DataFrame) -> pd.DataFrame:
    """Per-concern precision/recall/F1 against gold labels (column 'gold_concerns', '|'-joined)."""
    if "gold_concerns" not in df:
        return pd.DataFrame()
    gold = df["gold_concerns"].fillna("").apply(lambda s: set(s.split("|")) - {""})
    pred = df["concerns"].apply(set)
    rows = []
    for c in CONCERNS:
        tp = sum((c in g) and (c in p) for g, p in zip(gold, pred))
        fp = sum((c not in g) and (c in p) for g, p in zip(gold, pred))
        fn = sum((c in g) and (c not in p) for g, p in zip(gold, pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        rows.append({"concern": c, "precision": pr, "recall": rc, "f1": f1, "support": tp + fn})
    return pd.DataFrame(rows)


def _embed(texts):
    from sentence_transformers import SentenceTransformer  # optional dependency
    return SentenceTransformer("all-MiniLM-L6-v2").encode(list(texts), normalize_embeddings=True)


def cluster_feedback(texts, k=5, seed=0, method="tfidf", n_seeds=5):
    """Returns dict(labels, terms, silhouette, stability_ari, method)."""
    texts = list(texts)
    tv = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    T = tv.fit_transform(texts)
    X, used = T, "tfidf"
    if method == "embed":
        try:
            X, used = _embed(texts), "embed"
        except Exception:
            used = "tfidf (embed unavailable)"
    km = KMeans(k, n_init=10, random_state=seed).fit(X)
    labels = km.labels_
    sil = float(silhouette_score(X, labels, metric="cosine"))
    aris = [adjusted_rand_score(labels, KMeans(k, n_init=10, random_state=s).fit(X).labels_)
            for s in range(1, n_seeds + 1)]
    vocab = np.array(tv.get_feature_names_out())
    terms = {}
    for c in range(k):
        m = np.asarray(T[labels == c].mean(axis=0)).ravel()
        terms[c] = list(vocab[m.argsort()[::-1][:6]])
    return {"labels": labels, "terms": terms, "silhouette": sil,
            "stability_ari": float(np.mean(aris)), "method": used}
