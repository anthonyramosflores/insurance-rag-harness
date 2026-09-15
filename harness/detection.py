"""
Hallucination detection methods. Each returns a DetectionResult the harness
can use to decide whether to pass, retry, fall back, or route to a human.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import anthropic

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

_client = None


def _get_client():
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


@dataclass
class DetectionResult:
    method: str
    flagged: bool
    score: float  # 0.0 = fully trustworthy, 1.0 = fully suspect
    details: dict = field(default_factory=dict)


GROUNDEDNESS_PROMPT = """You are a strict fact-checker. Below is a CONTEXT (source
documents) and an ANSWER generated from that context. Break the answer into its
individual factual claims (numbers, dates, statuses, names, amounts). For each
claim, decide whether it is directly supported by the context.

Respond ONLY with JSON in this exact shape, no other text:
{{
  "claims": [
    {{"claim": "...", "supported": true, "reason": "..."}}
  ]
}}

--- CONTEXT ---
{context}
--- END CONTEXT ---

--- ANSWER ---
{answer}
--- END ANSWER ---"""


def groundedness_check(answer: str, context: str) -> DetectionResult:
    """Decompose the answer into claims and verify each against the context."""
    client = _get_client()
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": GROUNDEDNESS_PROMPT.format(context=context, answer=answer),
        }],
    )
    raw = message.content[0].text.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
        claims = parsed.get("claims", [])
    except (json.JSONDecodeError, AttributeError):
        return DetectionResult(
            method="groundedness",
            flagged=True,
            score=1.0,
            details={"error": "could_not_parse_checker_output", "raw": raw[:500]},
        )

    if not claims:
        return DetectionResult(method="groundedness", flagged=False, score=0.0, details={"claims": []})

    unsupported = [c for c in claims if not c.get("supported", True)]
    score = len(unsupported) / len(claims)

    return DetectionResult(
        method="groundedness",
        flagged=len(unsupported) > 0,
        score=round(score, 3),
        details={"claims": claims, "unsupported_count": len(unsupported), "total_claims": len(claims)},
    )


HEDGE_PHRASES = [
    "i don't know", "i'm not sure", "not available in the context",
    "cannot find", "could not find", "couldn't find", "no information",
    "unclear from the context", "not present in the context",
    "unable to determine", "does not contain", "doesn't contain",
    "no relevant", "not mentioned",
]


def confidence_signal(answer: str) -> DetectionResult:
    """
    Cheap heuristic run before the expensive checks: does the answer hedge
    appropriately, or cite a source like '[1]'? Absence of both is a mild
    yellow flag.
    """
    lower = answer.lower()
    hedges = any(phrase in lower for phrase in HEDGE_PHRASES)
    has_citation = bool(re.search(r"\[\d+\]", answer))

    flagged = not hedges and not has_citation
    return DetectionResult(
        method="confidence_signal",
        flagged=flagged,
        score=1.0 if flagged else 0.0,
        details={"hedged": hedges, "has_citation": has_citation},
    )


PARAPHRASE_PROMPT = """Rephrase the following question in a different way, preserving
its exact meaning and intent. Respond with ONLY the rephrased question, nothing else.

Question: {query}"""


def self_consistency_check(generate_fn, user_query: str, context: str) -> DetectionResult:
    """
    Ask the same underlying question two different ways and check whether
    the answers agree on their factual claims. Disagreement is a sign the
    model isn't reliably grounded -- it's reconstructing an answer each time
    rather than reading it consistently off the context.
    """
    client = _get_client()

    paraphrase_msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(query=user_query)}],
    )
    paraphrased_query = paraphrase_msg.content[0].text.strip()

    answer_a = generate_fn(user_query, context)
    answer_b = generate_fn(paraphrased_query, context)

    compare_prompt = (
        "Below are two answers to differently-worded versions of the same underlying "
        "question. Do they agree on all factual claims (numbers, dates, statuses, amounts)? "
        "Respond ONLY with JSON: {\"consistent\": true/false, \"disagreements\": [\"...\"]}\n\n"
        f"Answer A: {answer_a}\n\nAnswer B: {answer_b}"
    )
    compare_msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": compare_prompt}],
    )
    raw = re.sub(r"^```(json)?|```$", "", compare_msg.content[0].text.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
        consistent = parsed.get("consistent", False)
        disagreements = parsed.get("disagreements", [])
    except (json.JSONDecodeError, AttributeError):
        return DetectionResult(
            method="self_consistency",
            flagged=True,
            score=1.0,
            details={"error": "could_not_parse_checker_output", "answer_a": answer_a, "answer_b": answer_b},
        )

    return DetectionResult(
        method="self_consistency",
        flagged=not consistent,
        score=0.0 if consistent else 1.0,
        details={
            "paraphrased_query": paraphrased_query,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "disagreements": disagreements,
        },
    )