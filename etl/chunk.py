"""
Convert structured DataFrames into text chunks suitable for embedding.

Each chunk carries metadata so retrieved results (and any hallucination
flagged downstream) can be traced back to the exact source record.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

import pandas as pd

CHUNK_MAX_CHARS = 512


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def policy_to_chunk(row: pd.Series) -> Chunk:
    text = (
        f"Policy {row.get('policy_id', 'N/A')} is a {row.get('policy_type', 'N/A')} policy "
        f"for customer {row.get('customer_id', 'N/A')} in {row.get('state', 'N/A')}. "
        f"Status: {row.get('status', 'N/A')}. "
        f"Coverage: ${row.get('coverage_amount_usd', 0):,.0f}. "
        f"Annual premium: ${row.get('premium_usd', 0):,.0f}. "
        f"Effective {row.get('start_date', 'N/A')} through {row.get('end_date', 'N/A')}."
    )
    return Chunk(
        text=textwrap.shorten(text, width=CHUNK_MAX_CHARS, placeholder="..."),
        metadata={
            "source": "policies",
            "policy_id": row.get("policy_id"),
            "policy_type": row.get("policy_type"),
            "state": row.get("state"),
        },
    )


def claim_to_chunk(row: pd.Series) -> Chunk:
    text = (
        f"Claim {row.get('claim_id', 'N/A')} under policy {row.get('policy_id', 'N/A')} "
        f"filed on {row.get('claim_date', 'N/A')}. "
        f"Status: {row.get('status', 'N/A')}. "
        f"Amount claimed: ${row.get('amount_claimed_usd', 0):,.0f}. "
        f"Amount paid: ${row.get('amount_paid_usd', 0):,.0f}. "
        f"Details: {row.get('description', '')}."
    )
    return Chunk(
        text=textwrap.shorten(text, width=CHUNK_MAX_CHARS, placeholder="..."),
        metadata={
            "source": "claims",
            "claim_id": row.get("claim_id"),
            "policy_id": row.get("policy_id"),
            "status": row.get("status"),
        },
    )


def customer_to_chunk(row: pd.Series) -> Chunk:
    text = (
        f"Customer {row.get('customer_id', 'N/A')} ({row.get('name', 'N/A')}) "
        f"is located in {row.get('state', 'N/A')}, age {row.get('age', 'N/A')}. "
        f"Risk score: {row.get('risk_score', 'N/A')}."
    )
    return Chunk(
        text=textwrap.shorten(text, width=CHUNK_MAX_CHARS, placeholder="..."),
        metadata={
            "source": "customers",
            "customer_id": row.get("customer_id"),
            "state": row.get("state"),
        },
    )


_CHUNKERS = {
    "policies": policy_to_chunk,
    "claims": claim_to_chunk,
    "customers": customer_to_chunk,
}


def dataframe_to_chunks(name: str, df: pd.DataFrame) -> list[Chunk]:
    chunker = _CHUNKERS.get(name)
    if not chunker:
        raise ValueError(f"No chunker registered for '{name}'")
    return [chunker(row) for _, row in df.iterrows()]


def all_chunks(transformed: dict[str, pd.DataFrame]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for name, df in transformed.items():
        chunks.extend(dataframe_to_chunks(name, df))
    print(f"Generated {len(chunks)} total chunks across {len(transformed)} datasets.")
    return chunks