
from sentence_transformers import SentenceTransformer
MODEL=None
def encode(texts):
    global MODEL
    if MODEL is None: MODEL=SentenceTransformer("all-MiniLM-L6-v2")
    return MODEL.encode(list(texts),normalize_embeddings=True,show_progress_bar=True)
