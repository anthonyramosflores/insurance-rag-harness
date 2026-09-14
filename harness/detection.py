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
        # Fail safe, not fail open: if the checker itself is unparseable,
        # treat the answer as suspect rather than silently passing it.
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
    "cannot find", "no information", "unclear from the context",
    "not present in the context", "unable to determine",
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