"""
Retrieve relevant document chunks for a user query.
"""

from __future__ import annotations

from embeddings.generate import embed_texts
from embeddings.store import query as vector_query


def retrieve(
    user_query: str,
    n_results: int = 5,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Embed the user query and return the top-n matching document chunks.
    Returns: list of dicts with keys: text, metadata, distance.
    """
    query_vectors = embed_texts([user_query])
    query_vector = query_vectors[0]

    where = {"source": source_filter} if source_filter else None
    results = vector_query(query_vector, n_results=n_results, where=where)

    return results


def format_context(retrieved: list[dict]) -> str:
    """Concatenate retrieved chunks into a numbered context string for the LLM prompt."""
    if not retrieved:
        return "No relevant documents found."
    parts = []
    for i, item in enumerate(retrieved, 1):
        source = item["metadata"].get("source", "unknown")
        parts.append(f"[{i}] ({source}) {item['text']}")
    return "\n\n".join(parts)