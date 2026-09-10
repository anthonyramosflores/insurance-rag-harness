"""
Generate text embeddings using a local sentence-transformers model.
"""

from __future__ import annotations

import os

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return an embedding vector for each input text."""
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


def embed_chunks(chunks) -> list[tuple]:
    """
    Accepts a list of Chunk objects (from etl/chunk.py) and returns
    (text, embedding_vector, metadata) tuples ready for ChromaDB insertion.
    """
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    return [(c.text, vec, c.metadata) for c, vec in zip(chunks, vectors)]