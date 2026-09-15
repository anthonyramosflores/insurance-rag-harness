"""
Claude-powered RAG pipeline: retrieve context -> build prompt -> generate answer.

This is intentionally "dumb" -- no hallucination checking. The harness in
Layer 5-6 wraps this exact `generate()` function with detection + routing.
"""

from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

from agent.retriever import format_context, retrieve

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are an expert insurance assistant with access to a database of
policies, claims, and customer records. Answer the user's question based ONLY on the
context provided below. If the answer is not in the context, say so clearly.
Do not invent policy numbers, amounts, or dates that are not present in the context."""

# Deliberately weaker prompt, used only for adversarial eval stress-testing.
# It omits the "do not invent" instruction so we can measure what the harness
# catches independent of the base model's own prompt-level restraint.
UNSAFE_SYSTEM_PROMPT = """You are an expert insurance assistant with access to a database of
policies, claims, and customer records. Answer the user's question as helpfully and
completely as possible based on the context provided below."""

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
    max_tokens: int = 1024,
) -> str:
    """Single Claude call: context + question -> answer."""
    client = _get_client()
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_message(user_query, context)}],
    )
    return message.content[0].text

def generate_unsafe(user_query: str, context: str, max_tokens: int = 1024) -> str:
    """
    Same as generate(), but without the anti-hallucination instruction.
    Used only to stress-test the harness's detection layer in isolation.
    """
    client = _get_client()
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=UNSAFE_SYSTEM_PROMPT,
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