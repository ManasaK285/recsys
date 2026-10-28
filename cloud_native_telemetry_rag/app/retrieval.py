from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class FaissRetriever:
    def __init__(self, docs_dir, model_name):
        self.docs_dir = Path(docs_dir)
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.index = None
        self._load_documents()

    def _load_documents(self):
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted([*self.docs_dir.glob("*.md"), *self.docs_dir.glob("*.txt")])
        if not paths:
            raise RuntimeError(f"No .md or .txt documents found in {self.docs_dir}")
        for path in paths:
            content = path.read_text(encoding="utf-8").strip()
            chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
            for i, chunk in enumerate(chunks):
                self.documents.append({
                    "document_id": f"{path.stem}#{i}",
                    "title": path.stem.replace("_", " ").title(),
                    "text": chunk,
                })
        vectors = self.model.encode([d["text"] for d in self.documents],
                                    normalize_embeddings=True, convert_to_numpy=True)
        vectors = np.asarray(vectors, dtype="float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query, top_k=3):
        vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self.index.search(np.asarray(vector, dtype="float32"),
                                             min(top_k, len(self.documents)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({**self.documents[int(idx)], "score": float(score)})
        return results

    def evaluate_recall(self, eval_path, k=3):
        examples = json.loads(Path(eval_path).read_text(encoding="utf-8"))
        scores, detail = [], []
        for item in examples:
            relevant = set(item.get("relevant_document_ids", []))
            if not relevant:
                continue
            found = {r["document_id"].split("#")[0] for r in self.search(item["query"], k)}
            recall = len(found & relevant) / len(relevant)
            scores.append(recall)
            detail.append({"query": item["query"], "recall_at_k": recall,
                           "relevant": sorted(relevant), "retrieved": sorted(found)})
        return {"recall_at_k": sum(scores) / len(scores) if scores else 0.0,
                "k": k, "evaluated_queries": len(scores), "details": detail}
