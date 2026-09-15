"""
Adversarial eval: run the harness with the *unsafe* (no anti-hallucination
instruction) generator, to measure what the detection layer catches when
the base model isn't already guarding itself via the system prompt.

This isolates the harness's contribution from the LLM's own prompt-level
restraint -- a well-prompted model may already refuse to hallucinate, which
would make catch_rate meaningless if we only ever tested the safe prompt.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from agent.pipeline import generate_unsafe
from harness.eval_set import load_eval_set, run_eval

if __name__ == "__main__":
    examples = load_eval_set()
    # Swap the harness's generator for the unsafe one, for this run only.
    with patch("harness.harness.claude_generate", generate_unsafe):
        summary = run_eval(examples)

    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print()
    for r in summary["results"]:
        print("---")
        print("Label:", r["example"].label)
        print("Query:", r["example"].query)
        print("Route:", r["result"].route)
        print("Groundedness score:", r["result"].groundedness_score)
        print("Answer:", r["result"].answer[:250])
        