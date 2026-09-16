# Insurance RAG Agent + Hallucination Harness

A RAG pipeline over mock insurance data (policies/claims/customers), wrapped in a
hallucination detection and routing harness.

## Architecture

    ingestion/     Mock CSV data generation + loading
    etl/           Pandas cleaning + text chunking
    embeddings/    Local sentence-transformers embeddings -> ChromaDB
    agent/         Naive RAG: retrieve + Claude (Sonnet) generation, no checks
    harness/       detection.py (3 signals) + harness.py (routing) + eval_set.py
    evaluation/    SQLite run logging

`agent/pipeline.py` is the naive "before" picture: retrieve + generate, no
hallucination checking. `harness/harness.py` wraps it with three independent
detectors and routes each answer to PASS / RETRY / FALLBACK / HUMAN_REVIEW.

## Detection methods

- **`groundedness_check`** — a second Claude call decomposes the answer into
  individual factual claims and verifies each against the retrieved context.
- **`confidence_signal`** — a free heuristic: does the answer hedge
  appropriately, or cite a source? No API call required.
- **`self_consistency_check`** — paraphrases the query and checks whether the
  model's answer agrees with itself across the two phrasings.

## Eval methodology

`harness/eval_set.py` runs a labeled set of queries through the harness and
computes hallucination rate, catch rate (recall on labeled hallucination-prone
queries), and false-positive rate (on labeled clean queries).

`harness/adversarial_eval.py` runs the same eval set through a **deliberately
weakened system prompt** (`generate_unsafe` in `agent/pipeline.py`) that omits
the anti-hallucination instruction. This isolates the harness's own
contribution from the base model's prompt-level restraint — testing only
against a well-prompted model would understate what the harness actually adds,
since Claude Sonnet already refuses to hallucinate on many query types even
with no explicit warning against it.

## Known limitations

- **Groundedness false positives on procedural framing.** The groundedness
  checker verifies every claim in an answer against the retrieved context,
  including claims about the model's own process (e.g. "here's a randomly
  selected customer") rather than only claims about the underlying data. Since
  "randomness" isn't something the context can confirm or deny, this can
  trigger a false flag on an answer that is factually correct. The checker
  fails safe (over-flagging is lower-risk than under-flagging), but this is a
  real precision limitation worth fixing in a v2 — e.g. by prompting the
  checker to only evaluate claims about domain facts, not meta-commentary
  about how the answer was constructed.
- **Non-deterministic eval results.** Claude's outputs aren't fully
  deterministic between calls (sampling controls like `temperature` were
  removed from the Anthropic SDK's v1.0 release), so the same query can
  produce a different answer — and therefore a different groundedness
  verdict — across separate runs. A single eval pass is a noisy estimate, not
  a fixed ground truth. A more rigorous version would run each eval query
  multiple times and report an average flag rate with variance, rather than
  treating one run as definitive.
- **Small, hand-built eval set.** `data/eval_set.jsonl` currently has 5
  examples, enough to validate the harness's mechanics but not enough to
  produce statistically meaningful catch/false-positive rates. Scaling this
  to dozens of examples per category would be the next step for a genuinely
  trustworthy metric.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # add your ANTHROPIC_API_KEY

python ingestion/mock_data.py

python -c "
from ingestion.loader import load_all
from etl.transform import run_all_transforms
from etl.chunk import all_chunks
from embeddings.generate import embed_chunks
from embeddings.store import upsert_chunks

raw = load_all()
transformed = run_all_transforms(raw)
chunks = all_chunks(transformed)
embedded = embed_chunks(chunks)
upsert_chunks(embedded)
print('Indexed', len(chunks), 'chunks')
"

python -m harness.eval_set          # safe-prompt eval
python -m harness.adversarial_eval  # weakened-prompt adversarial eval
```