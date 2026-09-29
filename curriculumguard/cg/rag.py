"""Local RAG: TF-IDF retrieval over evidence/*.md with a retrieval self-check."""
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
CONCERN_QUERY = {
    "accessibility": "students with disabilities accessibility assistive technology",
    "privacy": "student data privacy protection children",
    "academic_integrity": "cheating academic integrity AI detectors",
    "language_access": "multilingual English learners translation language access",
    "teacher_workload": "teacher workload training time",
}


class Retriever:
    def __init__(self, folder=EVIDENCE):
        self.src, self.txt = [], []
        for f in sorted(Path(folder).glob("*.md")):
            for para in f.read_text(encoding="utf-8").split("\n\n"):
                if len(para) > 60 and not para.startswith("#"):
                    self.src.append(f.stem)
                    self.txt.append(para.strip())
        self.tv = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.X = self.tv.fit_transform(self.txt)

    def search(self, query, k=2):
        sims = cosine_similarity(self.tv.transform([query]), self.X)[0]
        return [{"source": self.src[i], "text": self.txt[i], "score": float(sims[i])} for i in np.argsort(sims)[::-1][:k]]

    def evaluate(self, k=2):
        """Hit@k: does the concern's own evidence file appear in top-k for that concern's query?"""
        hits = {c: any(r["source"] == c for r in self.search(q, k)) for c, q in CONCERN_QUERY.items()}
        return hits, float(np.mean(list(hits.values())))
