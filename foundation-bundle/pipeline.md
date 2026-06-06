# Build Spec: Grounded Thai RAG Pipeline (FahMai)

This is the **implementation spec** for the system, sitting between the concepts in
[`foundation.md`](./foundation.md) and the measurement plan in [`evalplan.md`](./evalplan.md).

## 0. Decisions of Record

These were settled before writing this spec. They constrain everything below.

| Decision | Choice | Why |
|---|---|---|
| Goal | Grounded Q&A over our data, with citations | Facts change; answers must be traceable to a source |
| Approach | **RAG-first** on an open model (Path A) | A new base model is not feasible on this budget/compute |
| Fine-tune | LoRA held in reserve (Path B), only if RAG + a good base model underperform on Thai | Avoid spend until evidence justifies it |
| Pretrain from scratch | **No** | ~450–500 GPU-days for even a 7B model; out of scope and unnecessary |
| Language | Thai-dominant | Drives tokenizer, embeddings, chunking, generation model |
| Data | ~600GB, sanitized + unsanitized, **contains PII** | PII filtering is a hard blocker, not optional |
| Compute | Colab Pro (dev) + RunPod on-demand (batch GPU) | No 24/7 GPU; spin up, run batch, spin down |
| Budget | Low (hundreds, not thousands, of $) | Dedup hard before the one-time embedding job; prefer open-source infra |

**Non-goal:** training on the 600GB. For RAG we *index* the corpus
(filter → dedup → chunk → embed → index); we do not run gradient updates on it.

---

## 1. Architecture

```text
600GB raw (Thai, sensitive)
 → 1. Quality filter + dedup        offline ─┐
 → 2. PII detection + redaction              │  built once,
 → 3. Chunking (Thai-aware)                  │  rebuilt only when
 → 4. Embedding (batch on RunPod)            │  corpus changes
 → 5. Index (vector + lexical)      ─────────┘
 ───────────────────────────────────────────
 → 6. Retrieval (hybrid) + rerank   online ─┐
 → 7. Generation (Thai LLM)                  │  per query
 → 8. Answer + citations            ─────────┘
 → 9. Eval harness  (drives 6–8 against the golden + attack sets)
```

Two halves with very different cost profiles:

- **Offline (1–5):** built once, rebuilt only when the corpus changes. The single
  largest one-time cost is **embedding the full corpus** (stage 4). Everything before
  it exists to shrink the corpus so that job is cheaper.
- **Online (6–8):** runs per query. Retrieval is cheap (CPU-friendly once vectors
  exist); generation is the only step that needs a GPU, and only while answering.

---

## 2. Stage 1 — Quality Filter + Dedup

Goal: shrink 600GB to only useful, unique text **before** the expensive stages.
Dedup also reduces privacy risk (memorized/duplicated PII) and improves retrieval
(no near-identical chunks crowding results).

```text
- Drop: empty/near-empty docs, boilerplate, navigation/markup junk, encoding errors
- Language gate: keep Thai (and intended mixed Thai/EN); drop off-target languages
- Exact dedup:    hash full documents (e.g. SHA-1 of normalized text)
- Near dedup:     MinHash / SimHash over shingles to catch near-duplicates
- Quality heuristics: length bounds, symbol/word ratio, repetition ratio
```

**Output:** a cleaned, deduped corpus with provenance kept per document
(`doc_id`, `source`, `original_path`, `sanitized: true/false`). Provenance is needed
later for citations and for auditing what got indexed.

**Sanitized vs unsanitized:** keep the flag through the pipeline. Unsanitized docs get
the same PII stage as everything else, but the flag lets us audit and, if needed,
quarantine them separately.

---

## 3. Stage 2 — PII Detection + Redaction  (HARD BLOCKER)

Sensitive data **must not** reach embeddings, the index, logs, or any external model.
Vectors leak PII too, and retrieved context is fed verbatim to the generator. So this
runs **before** chunking/embedding, and nothing downstream may see raw PII.

Two detector tiers:

**Tier A — structured PII (high precision, free, regex + validation):**

```text
Thai national ID   13 digits + checksum (validate the check digit, don't just regex)
Thai phone numbers  0X-XXXX-XXXX and variants
Email addresses
Credit card numbers Luhn check
URLs / IP addresses (policy-dependent)
```

**Tier B — names / addresses (NER):**

```text
Microsoft Presidio (open-source) as the orchestration layer
  + PyThaiNLP NER as a custom Thai recognizer for Thai names/locations
```

Tier A is high-precision and cheap; Tier B catches free-text names/addresses at lower
precision. Run both; redact (or hash, if a field must stay joinable) on any hit.

**Validation (this is the part people skip):** build a small labeled **Thai-PII test
set** and measure detector **recall** — for sensitive data, a miss is the costly error,
so optimize recall first, then precision. Treat this test set like the golden set:
versioned, held out. Do not ship until recall clears a stated threshold.

**Output:** redacted corpus + a redaction report (counts per type per source) for audit.

---

## 4. Stage 3 — Chunking (Thai-aware)

Thai has **no word spaces**, so whitespace splitting breaks. Segment first.

```text
Free text:  PyThaiNLP `newmm` word segmentation
            → group into chunks of ~256–512 tokens, ~15% overlap
            → keep doc_id + char offsets for citation back to source
Structured FahMai records:
            → chunk per record / per field, NOT as free text
            → keep field names so the generator can cite "field X of record Y"
```

Chunk size is a tuning knob, not a constant — it's a variant to sweep in the eval loop
(Section 9). Start at 256–512 tokens and let the golden set move it.

**Output:** chunks with `chunk_id`, `doc_id`, source offsets, and the `sanitized` flag.

---

## 5. Stage 4 — Embedding  (the one big GPU job)

**Model: `BAAI/bge-m3`.** Reasons: strong multilingual (good Thai), 8k context, and it
emits **dense + sparse** vectors in one pass. The sparse (lexical) signal is what keeps
recall up on Thai proper nouns, SKUs, and model numbers — the exact Thai-recall failure
mode flagged in the eval plan.

```text
Run as a batch job on RunPod (on-demand A100): spin up, embed, write vectors, spin down.
This is the largest one-time cost — which is why Stages 1–2 shrink the corpus first.
Persist vectors to disk/object storage so retrieval never needs the GPU again.
```

Embed only the redacted corpus. If the corpus changes, re-embed only the delta.

---

## 6. Stage 5 — Index

**Store: Qdrant** (open-source, self-host on a cheap CPU box) for native **hybrid**
search (dense + sparse from bge-m3) plus metadata filtering.

```text
Dev / golden-set scale:  Chroma or LanceDB locally (zero infra)
Full corpus:             Qdrant (hybrid + payload filters on doc/source/date)
Payload per point:       chunk_id, doc_id, source, offsets, dates, sanitized flag
```

Metadata filters matter for FahMai's bitemporal/date questions — they let retrieval
restrict by record type or date range before/after vector search.

---

## 7. Stage 6 — Retrieval + Rerank (online)

```text
Query → (optional) query understanding / planning
      → hybrid search in Qdrant (dense + sparse), top-k
      → rerank with BAAI/bge-reranker-v2-m3 → top-n
      → assemble context (with citations) for the generator
```

**Tools, not vectors, for exact logic** (from the eval plan):

- **Buddhist-era ↔ Gregorian dates** (BE = CE + 543): a deterministic conversion tool,
  including Thai numerals and partial dates. Never let the LLM "estimate" this.
- **Bitemporal "as-of" lookups** (valid time vs. system/transaction time): a DB-lookup
  tool that takes an as-of date and returns the record valid at that time.

`k`, `n`, and whether to rerank are eval variants, not fixed values.

---

## 8. Stage 7 — Generation (Thai LLM)

**Model: Typhoon (SCB 10X, purpose-built for Thai) or Qwen2.5-7B/14B-Instruct**
(very strong Thai). Pick via the golden set; both are open and self-hostable.

```text
Serve with vLLM on RunPod on-demand — spin up for batch/eval/production runs, spin down.
Do NOT keep a GPU running 24/7 on a low budget.
```

Generation rules baked into the system prompt + checked by eval:

```text
- Answer ONLY from retrieved context; cite the source for every claim
- If the answer is not in the data, say so — do not fabricate (refusal accuracy)
- Use the date/bitemporal tools for any date or "as-of" reasoning
```

**Privacy note:** because the corpus is PII-redacted, retrieved context is safe to feed
the model. *Queries* may still contain PII — keep generation **self-hosted** (Typhoon/
Qwen on RunPod), not a third-party API, unless/until a data-handling agreement says
otherwise.

---

## 9. Stage 9 — Evaluation Hookup

Eval is defined in [`evalplan.md`](./evalplan.md); this pipeline plugs into it directly.
Three axes: **accuracy**, **cost efficiency**, **adversarial/prompt-injection defense**.

```text
Build the golden set (100–200 items) BEFORE tuning — baseline first, fly blind never.
Score retrieval (Recall@k, nDCG) and answers (exact/faithfulness, refusal) separately.
Log tokens/cost/latency on every run — choose a point on the quality-vs-cost frontier.
Run the attack set separately; track attack success rate AND over-refusal.
Change one variable at a time; a 1–2 pt move is noise — re-run 3+ times.
```

Injection defense is especially live here because attacks can hide **inside retrieved
documents** (product descriptions, reviews). The redaction stage does not catch those —
the attack set does.

---

## 10. Hardware Mapping

| Where | Stages | Notes |
|---|---|---|
| **Colab Pro** | 1, 2, 3, dev, golden-set eval | CPU/light-GPU work; PII pipeline; experiments |
| **RunPod (on-demand A100)** | 4 (embed once), 7/8 (LLM batch) | Spin up → run → spin down. Biggest spend = stage 4 |
| **Cheap CPU box / local** | 5, 6 (retrieval serving) | Qdrant + hybrid search once vectors exist |

**Cost discipline:** dedup hard (Stage 1) → smaller embed job (Stage 4) → never idle a
GPU (Stages 7–8 on-demand). Those three rules are most of the budget.

---

## 11. Build Order (what to write first)

1. **PII filter module** + Thai-PII test set — the hard blocker. Measure recall; don't
   ship sensitive data until it clears threshold. ([Stage 2](#3-stage-2--pii-detection--redaction--hard-blocker))
2. **Golden set** (100–200 items) per the eval plan — needed before any tuning.
3. **Baseline RAG** end to end (bge-m3 + Qdrant hybrid + reranker + Typhoon/Qwen) and
   run it through all three eval axes.
4. **Iterate** one variable at a time: rerank on/off, chunk size, `k`/`n`, then LoRA
   only if the data says the base model is the bottleneck.

---

## 12. Open Items / Risks

```text
- Corpus size after dedup+filter is unknown → it sets the real embedding cost. Measure early.
- Thai-PII NER recall is the riskiest unknown → invest in the test set first.
- Mixed Thai/English or numerals-in-Thai may dent retrieval recall → cover in golden set.
- Injection inside retrieved docs is not a redaction problem → owned by the attack set.
- Self-hosted vs API generation hinges on query-PII policy → decide before any external call.
```
