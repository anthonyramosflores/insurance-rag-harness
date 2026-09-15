"""
Labeled eval set + runner for measuring the harness's effectiveness.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from harness.harness import Route, run as run_harness

EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_set.jsonl")


@dataclass
class EvalExample:
    query: str
    source_filter: str | None
    label: str  # "hallucination_prone" or "clean"


def load_eval_set(path: str = EVAL_SET_PATH) -> list[EvalExample]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(EvalExample(
                query=row["query"],
                source_filter=row.get("source_filter"),
                label=row["label"],
            ))
    return examples


def run_eval(examples: list[EvalExample]) -> dict:
    """Run the harness over every example and compute summary metrics."""
    results = []
    for ex in examples:
        result = run_harness(ex.query, source_filter=ex.source_filter, label=ex.label, log=True)
        flagged = result.route != Route.PASS
        results.append({"example": ex, "result": result, "flagged": flagged})

    hallucination_prone = [r for r in results if r["example"].label == "hallucination_prone"]
    clean = [r for r in results if r["example"].label == "clean"]

    caught = sum(1 for r in hallucination_prone if r["flagged"])
    false_positives = sum(1 for r in clean if r["flagged"])

    catch_rate = caught / len(hallucination_prone) if hallucination_prone else None
    false_positive_rate = false_positives / len(clean) if clean else None
    hallucination_rate = sum(1 for r in results if r["flagged"]) / len(results) if results else 0.0

    return {
        "n_examples": len(results),
        "hallucination_rate": round(hallucination_rate, 3),
        "catch_rate": round(catch_rate, 3) if catch_rate is not None else None,
        "false_positive_rate": round(false_positive_rate, 3) if false_positive_rate is not None else None,
        "results": results,
    }


if __name__ == "__main__":
    if not os.path.exists(EVAL_SET_PATH):
        print(f"No eval set found at {EVAL_SET_PATH}.")
    else:
        summary = run_eval(load_eval_set())
        print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))