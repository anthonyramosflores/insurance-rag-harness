"""
Persist and query embeddings in a local ChromaDB collection.
"""

from __future__ import annotations

import os

CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "chroma"),
)
COLLECTION_NAME = "insurance_docs"

_client = None


def get_client():
    global _client
    if _client is None:
        import chromadb
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client


def get_or_create_collection(client):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(embedded_chunks: list[tuple]) -> None:
    """Insert or update (text, vector, metadata) tuples into ChromaDB."""
    client = get_client()
    collection = get_or_create_collection(client)

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, (text, vector, meta) in enumerate(embedded_chunks):
        source = meta.get("source", "unknown")
        record_id = meta.get("policy_id") or meta.get("claim_id") or meta.get("customer_id") or str(i)
        ids.append(f"{source}::{record_id}::{i}")
        embeddings.append(vector)
        documents.append(text)
        metadatas.append(meta)

    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"Upserted {len(ids)} chunks into collection '{COLLECTION_NAME}'")


def query(query_vector: list[float], n_results: int = 5, where: dict | None = None) -> list[dict]:
    """Return the top-n most similar documents for a given query embedding."""
    client = get_client()
    collection = get_or_create_collection(client)

    kwargs = {"query_embeddings": [query_vector], "n_results": n_results}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(docs, metas, distances)
    ]