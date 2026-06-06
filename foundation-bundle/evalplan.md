# Evaluation Plan: Agentic RAG Pipeline (FahMai)

A practical eval plan for an agentic retrieval pipeline over the FahMai (Thai electronics retailer) dataset, scored on three axes: **accuracy**, **cost efficiency**, and **adversarial / prompt-injection defense**.

The goal is a repeatable benchmark you own, not a one-off score. Treat your golden set as the primary signal and any public benchmark as secondary.

---

## 1. What We Are Evaluating

Evaluate the **whole system**, not just the model:

```text
User query
→ Query understanding / planning
→ Retrieval (chunking, ranking, recall)
→ Tool calls (DB lookups, date conversion, etc.)
→ Answer generation
→ Final answer + citations
```

Score the model and the system separately where possible, because a faithful model given bad retrieval still produces a wrong answer. Retrieval quality and generation quality are different failure modes and need different metrics.

---

## 2. The Golden Set (Primary Benchmark)

Build a curated, labeled set drawn from real FahMai data. Aim for **100–200 examples** to start; enough to separate candidates, small enough to label well and re-run cheaply.

Each item should record:

```text
id
query (Thai and/or English)
query_category (see below)
retrieved_truth (which records/chunks SHOULD be retrieved)
expected_answer (gold answer or acceptance criteria)
required_tool_calls (e.g. date conversion, bitemporal lookup)
difficulty (easy / medium / hard)
notes (edge cases, why it is tricky)
```

Cover the categories that actually stress the system:

```text
Simple factual lookup
Multi-hop / multi-record reasoning
Thai-language queries
Buddhist-era ↔ Gregorian date handling
Bitemporal "as-of" questions (valid time vs. system/transaction time)
Aggregation / counting
Questions with NO answer in the data (should refuse, not hallucinate)
Ambiguous queries needing clarification
```

Keep the golden set **versioned and held out** from any prompt/example tuning, so you are not measuring contamination of your own making.

---

## 3. Axis 1 — Accuracy

Separate retrieval accuracy from answer accuracy.

**Retrieval metrics** (against `retrieved_truth`):

```text
Recall@k     – did the right records appear in the top k?
Precision@k  – how much retrieved context was relevant?
MRR / nDCG   – ranking quality of the right records
```

**Answer metrics** (against `expected_answer`):

```text
Exact match / normalized match – for factual + numeric answers
Faithfulness / groundedness     – is every claim supported by retrieved context?
Citation correctness            – do cited sources actually contain the claim?
Refusal accuracy                – does it correctly say "not in the data" when true?
```

Notes on judging:
- Use **exact/programmatic checks** for anything numeric, date-based, or lookup-style. Do not let an LLM judge grade what a string comparison can grade.
- Use an **LLM-as-judge** only for open-ended faithfulness, and validate it against ~20 human-labeled items first. Watch for verbosity and self-preference bias.

---

## 4. Axis 2 — Cost Efficiency

Record cost and latency on **every** golden-set run so you always see quality and cost together.

```text
Input tokens (incl. retrieved context)
Output tokens
Tool-call count
$ per query (model + tool/infra)
End-to-end latency (p50, p95)
```

Decision rule: do not pick a single "best" model. Plot accuracy vs. cost and choose a point on the frontier that clears your accuracy threshold at acceptable cost. A small accuracy gain that doubles cost is usually the wrong trade for a competition with a cost-efficiency criterion.

Cheap levers to test as variants: smaller retrieval `k`, tighter context, a cheaper model for routing/planning with a stronger model only for the final answer, and caching.

---

## 5. Axis 3 — Adversarial / Prompt-Injection Defense

This is a **separate eval set** with its own metric. Accuracy benchmarks will not reveal these failures.

Build an attack set covering:

```text
Injected instructions inside retrieved documents ("ignore previous instructions...")
Instructions hidden in data fields, product descriptions, or reviews
Attempts to exfiltrate the system prompt or other records
Attempts to trigger unauthorized tool calls
Role-play / jailbreak framings
Multi-step / delayed injections across turns
```

Primary metric:

```text
Attack success rate (lower is better)
  = injected attacks that changed behavior / total attacks
```

Also track **over-refusal**: a defense that blocks legitimate queries is also a failure. Measure false-positive refusals on clean queries so you do not trade security for usability blindly.

---

## 6. Domain-Specific Checks (FahMai)

These are easy to get wrong and worth dedicated golden-set items:

- **Buddhist-era dates**: BE = CE + 543. Test both directions, partial dates, and dates written in Thai numerals. Confirm the pipeline converts consistently and does not silently mix calendars.
- **Bitemporal logic**: distinguish *valid time* (when a fact was true in the real world) from *system/transaction time* (when it was recorded). Test "as of" questions where the two diverge — e.g. a price that changed, queried as of a past date.
- **Thai language**: tokenization, no word spaces, mixed Thai/English queries, and Thai-specific formatting. Verify retrieval recall does not collapse on Thai-only queries.
- **No-answer cases**: confirm the system refuses or says "not found" rather than fabricating, especially for plausible-sounding but absent records.

---

## 7. Methodology and Reproducibility

```text
Pin the exact model version/string (behavior drifts behind the same name)
Fix seeds and temperature; record them
Run each config 3+ times; report mean and a spread/CI, not a single number
Keep the harness identical across all candidates (prompt, parsing, k)
Version the golden set and attack set
Log every run: inputs, outputs, retrieved context, tokens, cost, latency
```

A 1–2 point difference between configs is likely noise. Only act on gaps that survive repeated runs.

---

## 8. Process

1. **Baseline**: run the current pipeline on the golden set + attack set. Record all three axes.
2. **Iterate**: change one thing at a time (retrieval, prompt, model, defense). Re-run.
3. **Regression guard**: keep the golden set in CI so a fix for one category does not silently break another.
4. **Decide**: pick the config that clears the accuracy and security thresholds at the best cost/latency point on the frontier.

---

## 9. Summary Scorecard (per config)

| Axis | Metric | Target |
|---|---|---|
| Retrieval | Recall@k | (set threshold) |
| Answer | Exact / faithfulness | (set threshold) |
| Answer | Refusal accuracy | (set threshold) |
| Cost | $ per query | (set budget) |
| Cost | Latency p95 | (set budget) |
| Security | Attack success rate | as low as possible |
| Security | False-positive refusals | low |

Fill the targets from the competition's stated thresholds, then treat any config that fails a hard threshold as disqualified regardless of how good its other numbers look.
