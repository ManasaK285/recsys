import re
import numpy as np

def text_features(text):
    """Transparent baseline text features, not a semantic-quality evaluator."""
    text = text or ""
    words = re.findall(r"\b\w+\b", text.lower())
    n = max(len(words), 1)
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    return np.asarray([
        np.log1p(n), np.log1p(sentences), len(set(words))/n,
        sum(map(len, words))/n/10, float("?" in text),
        float(any(w in text.lower() for w in ["because", "therefore", "example"])),
        np.log1p(text.count("\n\n")+1),
        float(any(w in text.lower() for w in ["please", "thanks", "sorry"]))
    ], dtype=np.float32)
