# Foundation-F1 — Thai 1B Foundation Model Pipeline

Builds a **~1B-parameter Thai foundation model** by using **`google/gemma-4-E2B-it`**
as an architecture + tokenizer template, deriving a smaller model, and pretraining
it from scratch on Thai corpora.

This is the **pretrain-a-base-model** path. It is deliberately separate from the
RAG-first plan in [`../foundation-bundle/`](../foundation-bundle/), which serves a
different goal (grounded Q&A over existing data without training a new base).

```
00_download_base  →  01_prune_to_1b  →  02_prepare_data  →  03_pretrain  →  04_eval
   tokenizer+         ~1B model           tokenize + pack     next-token      perplexity
   config             (reinit/structured) Thai corpora        pretraining     + samples
```

## What "prune to 1B then train without weights" means here

The request had a subtlety: if you discard the weights, "pruning" reduces to
**choosing a smaller architecture**. Stage 01 supports both readings via
`prune.mode` in `config.yaml`:

| mode | what it does | when to use |
|---|---|---|
| `reinit` *(default)* | Build a fresh dense ~1B text-decoder config from the Gemma 4 architecture family, **randomly initialise** the weights, reuse the tokenizer. A true from-scratch 1B. | You literally want "a 1B model without weights, trained from scratch." |
| `structured` | Load Gemma 4 E2B's pretrained text decoder and **prune it to ~1B** (drop evenly-spaced layers + trim MLP width), keeping inherited weights as a warm start. | You want a 1B model but better quality per training FLOP. Recommended in practice. |

A note on the base: `google/gemma-4-E2B-it` is the **E2B MatFormer** variant
(~2.3B effective / 5.1B with embeddings, 128K context, multimodal). We use only
its **text** config + tokenizer; the derived model is a dense text-only decoder.

## Data

Configured in `config.yaml` under `data`:

- `SPAISS6F1/spai-ss6-pythainlp-pretrain-collection` (HF)
- `SPAISS6F1/spai-ss6-llm-1b-thai-corpus` (HF)
- Local parquet already in this repo: `../data/ThaiLLMRepo_parquet/*.parquet`
  (schema `{text, source}`). SEA-PILE is available too — uncomment it in config.

All sources are normalised to a single `text` column, tokenized with the Gemma
tokenizer, and **packed** into fixed `sequence_length` blocks (no padding waste).

## Setup

```bash
cd pipeline
pip install -r requirements.txt

# Gemma is gated: accept the license on the model page, then authenticate.
export HF_TOKEN=hf_xxx            # or: huggingface-cli login
```

## Run

```bash
make base      # 00 download tokenizer + config
make prune     # 01 derive the ~1B model (mode set in config.yaml)
make data      # 02 build the tokenized dataset
make train     # 03 pretrain          (single GPU)
make eval      # 04 perplexity + samples

# or everything:  make all
# multi-GPU:       make train-dist N=8
```

## Apple Silicon (M2 Pro)

The pipeline auto-detects the backend: **CUDA → bf16**, **Apple MPS → fp32**, CPU → fp32.
There's a ready-made M2 profile, [`config.m2.yaml`](config.m2.yaml):

```bash
make base                       # download tokenizer/config (needs HF_TOKEN)
make m2                         # = make CONFIG=config.m2.yaml all
```

It's been verified end-to-end on an M2 Pro (MPS): prune → tokenize → train → eval
all run. What the profile changes for Apple Silicon:

- **Smaller model (~350M, not 1B).** A true 1B from-scratch pretrain does **not**
  fit an M2 Pro — fp32 Adam on 1B params is ~16 GB (params+grads+optimizer state)
  before activations, and MPS has no bf16/8-bit training to shrink it. 350M trains
  on 16–32 GB unified memory; treat it as the dev/iteration model and run the real
  1B on a CUDA GPU with `config.yaml`.
- **fp32** (forced on MPS by the scripts), **gradient checkpointing on**, tiny
  `per_device_train_batch_size` with large `gradient_accumulation_steps`, shorter
  `sequence_length` (1024), `dataloader_num_workers: 0` (MPS + workers can stall),
  and a `max_train_docs` cap so iteration stays fast.

The Makefile exports `PYTORCH_ENABLE_MPS_FALLBACK=1` so any op MPS doesn't
implement falls back to CPU instead of crashing.

> **Env note:** if you hit `ImportError: cannot import name 'HybridCache'`, your
> `peft` is too old for transformers 5.x — `pip install -U "peft>=0.18"`.
> (Already pinned in `requirements.txt`.)

## Smoke test (cheap dry run before committing GPU hours)

Edit `config.yaml`:

```yaml
prune:   { arch: { num_hidden_layers: 4, hidden_size: 512 } }
data:    { max_train_docs: 2000, sequence_length: 512 }
train:   { max_steps: 50, save_steps: 50, eval_steps: 50,
           per_device_train_batch_size: 2, gradient_accumulation_steps: 1 }
```

Then `make prune data train eval`. This verifies every stage wires together end
to end on CPU/a small GPU in minutes. Revert for the real run.

## Compute reality check

Pretraining a 1B model to a useful state is **not** a laptop job — budget on the
order of tens-to-hundreds of billions of tokens and multi-GPU-days (A100/H100).
`structured` mode reaches a usable model far faster than `reinit` because it
starts from Gemma's learned representations. Tune `max_steps`, batch size, and
`gradient_accumulation_steps` to your hardware; use `train.resume` for spot/preemptible
instances.

## Outputs

```
artifacts/base              base tokenizer + config
artifacts/model-1b-init     the derived ~1B model + prune_summary.json
artifacts/tokenized         packed train/validation dataset
artifacts/checkpoints       training checkpoints + checkpoints/final
```

## After pretraining

A pretrained base only continues text; it does not follow instructions. Next
steps (out of scope here, see `../foundation-bundle/foundation.md`): SFT /
instruction tuning → preference tuning → safety tuning → the evaluation in
`../foundation-bundle/evalplan.md`.
