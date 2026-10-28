import logging
from app.config import settings

logger = logging.getLogger("rag")

def estimate_tokens(text):
    """Rough word-based estimate; use provider tokenizer/usage for exact values."""
    return max(1, len(text.split()))

class RAGEngine:
    def __init__(self, retriever):
        self.retriever = retriever
        self.vertex_client = None
        if settings.use_vertex:
            try:
                from google import genai
                self.vertex_client = genai.Client(
                    vertexai=True, project=settings.google_cloud_project,
                    location=settings.google_cloud_location)
            except Exception:
                logger.exception("Vertex AI client unavailable; using local fallback")

    def answer(self, question, top_k=3):
        retrieved = self.retriever.search(question, top_k)
        context = "\n\n".join(
            f"[{d['title']} | {d['document_id']}] {d['text']}" for d in retrieved)
        prompt = ("Answer using only the context. If the answer is absent, say so. "
                  "Cite source document IDs.\n\nCONTEXT:\n" + context +
                  "\n\nQUESTION: " + question)
        answer = self._fallback(retrieved)
        mode = "extractive_fallback"
        if self.vertex_client is not None:
            try:
                response = self.vertex_client.models.generate_content(
                    model=settings.vertex_model, contents=prompt)
                if getattr(response, "text", None):
                    answer, mode = response.text, "vertex_ai"
            except Exception:
                logger.exception("Vertex generation failed; using local fallback")
        return answer, retrieved, estimate_tokens(prompt), estimate_tokens(answer), mode

    @staticmethod
    def _fallback(retrieved):
        if not retrieved:
            return "I could not find relevant context in the local knowledge base."
        excerpts = "\n".join(
            f"- {d['text'][:700]} (source: {d['document_id']})" for d in retrieved)
        return ("Retrieved context (extractive fallback, not a generated synthesis):\n" + excerpts)
