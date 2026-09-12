"""
Claude-powered RAG pipeline: retrieve context -> build prompt -> generate answer.

This is intentionally "dumb" -- no hallucination checking. The harness in
Layer 5-6 wraps this exact `generate()` function with detection + routing.
"""

from __future__ import annotations

import os

import anthropic

from agent.retriever import format_context, retrieve

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are an expert insurance assistant with access to a database of
policies, claims, and customer records. Answer the user's question based ONLY on the
context provided below. If the answer is not in the context, say so clearly.
Do not invent policy numbers, amounts, or dates that are not present in the context."""

MAX_CONTEXT_CHUNKS = 5

_client = None


def _get_client():
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _build_user_message(user_query: str, context: str) -> str:
    return f"""--- RETRIEVED CONTEXT ---
{context}
--- END CONTEXT ---

User question: {user_query}"""


def generate(
    user_query: str,
    context: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """Single Claude call: context + question -> answer."""
    client = _get_client()
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_message(user_query, context)}],
    )
    return message.content[0].text


def answer(
    user_query: str,
    n_results: int = MAX_CONTEXT_CHUNKS,
    source_filter: str | None = None,
) -> dict:
    """Full naive RAG pipeline: retrieve -> prompt -> generate."""
    retrieved = retrieve(user_query, n_results=n_results, source_filter=source_filter)
    context = format_context(retrieved)
    answer_text = generate(user_query, context)

    return {"answer": answer_text, "sources": retrieved, "context": context}