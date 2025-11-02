# src/utils.py
from sentence_transformers import SentenceTransformer
import numpy as np

_MODEL = None

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    return _MODEL

def embed_texts(texts, model=None):
    """
    texts: list[str] -> returns (n, dim) numpy array normalized for cosine-sim
    """
    if model is None:
        model = load_embedding_model()
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms
