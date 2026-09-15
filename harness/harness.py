"""
The harness: query -> generate -> detect -> route.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from agent.pipeline import generate as claude_generate
from agent.retriever import format_context, retrieve
from harness.detection import confidence_signal, groundedness_check, self_consistency_check

GROUNDEDNESS_RETRY_THRESHOLD = float(os.getenv("GROUNDEDNESS_RETRY_THRESHOLD", "0.15"))
GROUNDEDNESS_FALLBACK_THRESHOLD = float(os.getenv("GROUNDEDNESS_FALLBACK_THRESHOLD", "0.4"))
RUN_SELF_CONSISTENCY = os.getenv("RUN_SELF_CONSISTENCY", "true").lower() == "true"

FALLBACK_MESSAGE = (
    "I don't have enough verified information in the retrieved records to answer "
    "this confidently. Please rephrase the question or have a specialist review it."
)


class Route(str, Enum):
    PASS = "pass"
    RETRY = "retry"
    FALLBACK = "fallback"
    HUMAN_REVIEW = "human_review"


@dataclass
class HarnessResult:
    query: str
    answer: str
    route: Route
    sources: list = field(default_factory=list)
    groundedness_score: float | None = None
    self_consistency_flagged: bool | None = None
    confidence_flagged: bool | None = None
    detail: dict = field(default_factory=dict)


def run(
    user_query: str,
    n_results: int = 5,
    source_filter: str | None = None,
) -> HarnessResult:
    """Run the full harnessed pipeline for one query."""
    retrieved = retrieve(user_query, n_results=n_results, source_filter=source_filter)
    context = format_context(retrieved)

    answer_text = claude_generate(user_query, context)

    conf = confidence_signal(answer_text)
    ground = groundedness_check(answer_text, context)

    route = Route.PASS
    detail = {"confidence": conf.details, "groundedness": ground.details}

    if ground.score >= GROUNDEDNESS_FALLBACK_THRESHOLD:
        answer_text = FALLBACK_MESSAGE
        route = Route.FALLBACK
    elif ground.score >= GROUNDEDNESS_RETRY_THRESHOLD:
        retry_answer = claude_generate(user_query, context)
        retry_ground = groundedness_check(retry_answer, context)
        detail["retry_groundedness"] = retry_ground.details
        if retry_ground.score >= GROUNDEDNESS_RETRY_THRESHOLD:
            answer_text = FALLBACK_MESSAGE
            route = Route.FALLBACK
        else:
            answer_text = retry_answer
            ground = retry_ground
            route = Route.RETRY

    self_consistency_flagged = None
    if RUN_SELF_CONSISTENCY and route == Route.PASS:
        consistency = self_consistency_check(claude_generate, user_query, context)
        self_consistency_flagged = consistency.flagged
        detail["self_consistency"] = consistency.details
        if consistency.flagged:
            route = Route.HUMAN_REVIEW

    return HarnessResult(
        query=user_query,
        answer=answer_text,
        route=route,
        sources=retrieved,
        groundedness_score=ground.score,
        self_consistency_flagged=self_consistency_flagged,
        confidence_flagged=conf.flagged,
        detail=detail,
    )