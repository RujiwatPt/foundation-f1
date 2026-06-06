"""Generates thai_1b_t4.ipynb (self-contained Colab T4 pipeline).

Run:  python _build_notebook.py
Keeps notebook JSON valid by construction (nbformat-style dict -> json).
"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": text.strip("\n").splitlines(keepends=True),
    })


# ---------------------------------------------------------------- title
md("""# Thai 1B Foundation Model — Colab T4 Pipeline

Builds a **~1B-parameter Thai foundation model** from **`google/gemma-4-E2B-it`**
(architecture + tokenizer template) and pretrains it from scratch on:

- `SPAISS6F1/spai-ss6-pythainlp-pretrain-collection`
- `SPAISS6F1/spai-ss6-llm-1b-thai-corpus`

**Runtime → Change runtime type → T4 GPU** before running.

T4 notes baked in: **fp16** (Turing has no bf16), **SDPA** attention (no
FlashAttention-2 on T4), **8-bit paged AdamW** + **gradient checkpointing** to fit
the optimizer in 16 GB VRAM. Checkpoints can be written to Google Drive so you can
resume across Colab sessions.
""")

# ---------------------------------------------------------------- GPU check
md("## 1. Check the GPU")
code("""
import torch, subprocess
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
assert torch.cuda.is_available(), "No GPU. Runtime -> Change runtime type -> T4 GPU"
cap = torch.cuda.get_device_capability()
BF16_OK = cap[0] >= 8          # Ampere+; T4 is (7,5) -> False -> use fp16
print(f"CUDA capability {cap} | bf16 supported: {BF16_OK} | -> training dtype: {'bf16' if BF16_OK else 'fp16'}")
""")

# ---------------------------------------------------------------- install
md("## 2. Install dependencies")
code("""
%pip install -q -U "transformers>=4.46" "datasets>=2.20" accelerate "peft>=0.18" \\
    bitsandbytes sentencepiece huggingface_hub
print("done — if Colab asks, Runtime -> Restart session, then continue from cell 3.")
""")

# ---------------------------------------------------------------- login
md("""## 3. Authenticate with Hugging Face

Gemma is gated: accept the license once at
https://huggingface.co/google/gemma-4-E2B-it then log in (token needs *read*).""")
code("""
from huggingface_hub import login, whoami
login()  # paste a read token, or set HF_TOKEN as a Colab secret
print("Logged in as:", whoami()["name"])
""")

# ---------------------------------------------------------------- config
md("## 4. Configuration\nEdit here to scale size / data / steps.")
code("""
CONFIG = {
    "base_model": "google/gemma-4-E2B-it",   # architecture + tokenizer template

    # Target model size. ~1B fits a T4 with 8-bit optim + checkpointing (tight).
    # If you hit CUDA OOM, drop to 700_000_000 or lower hidden_size below.
    "target_params": 1_000_000_000,
    "arch": {
        "hidden_size": 1536,
        "intermediate_size": 6144,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "max_position_embeddings": 8192,
        "tie_word_embeddings": True,   # critical: shares the big vocab matrix
    },

    "datasets": [
        ("SPAISS6F1/spai-ss6-pythainlp-pretrain-collection", "train", "text"),
        ("SPAISS6F1/spai-ss6-llm-1b-thai-corpus", "train", "text"),
    ],
    "sequence_length": 1024,        # T4-friendly context for training

    # STREAMING (recommended for large corpora): pull shards on the fly instead of
    # downloading the whole repo to Colab disk first. Bounded by max_steps below.
    "streaming": True,
    "eval_blocks": 200,             # streaming: held-out packed blocks for eval
    "shuffle_buffer": 10_000,       # streaming shuffle buffer size

    # Non-streaming only (ignored when streaming=True):
    "max_train_docs": 200_000,      # cap so full download/tokenize fits Colab; None = all
    "validation_fraction": 0.002,

    # Training
    "output_dir": "/content/ckpt-thai-1b",   # set to a Drive path to persist
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,        # effective batch = 16
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_ratio": 0.01,
    "max_steps": 2000,              # raise for a real run; 2000 ~ a quick demo
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 500,
    "seed": 42,
}
print("config set. target:", f"{CONFIG['target_params']:,} params")
""")

# ---------------------------------------------------------------- build model
md("""## 5. Build the ~1B model (reinit / from scratch)

Reuses Gemma 4's tokenizer + architecture family, searches depth to hit the param
budget (counted on a **meta device** so it's exact for Gemma 4's per-layer
embeddings / AltUp blocks — a naive formula undercounts by ~2x), then random-inits.""")
code("""
import logging, torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
logging.getLogger("transformers").setLevel(logging.ERROR)

base_id = CONFIG["base_model"]
tok = AutoTokenizer.from_pretrained(base_id)
base_conf = AutoConfig.from_pretrained(base_id)
text_conf = getattr(base_conf, "text_config", base_conf)
vocab = text_conf.vocab_size
a = CONFIG["arch"]; target = CONFIG["target_params"]

def make_cfg(layers):
    kw = dict(vocab_size=vocab, hidden_size=a["hidden_size"],
              intermediate_size=a["intermediate_size"], num_hidden_layers=layers,
              num_attention_heads=a["num_attention_heads"],
              num_key_value_heads=a["num_key_value_heads"],
              max_position_embeddings=a["max_position_embeddings"],
              tie_word_embeddings=a["tie_word_embeddings"])
    try:
        return text_conf.__class__(**kw)
    except Exception:
        from transformers import Gemma3TextConfig
        return Gemma3TextConfig(**kw)

def meta_count(layers):
    c = make_cfg(layers)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(c)
    return sum(p.numel() for p in m.parameters()), c

best, prev = None, None
for L in range(1, 60):
    n, c = meta_count(L)
    if best is None or abs(n-target) < best[0]:
        best = (abs(n-target), L, n, c)
    if prev is not None and n > target:
        break
    prev = n
_, layers, n_est, conf = best
print(f"Selected {layers} layers -> ~{n_est/1e6:.1f}M params (target {target/1e6:.0f}M)")

model = AutoModelForCausalLM.from_config(conf, attn_implementation="sdpa")
print(f"Materialised: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
""")

# ---------------------------------------------------------------- data
md("""## 6. Load + tokenize + pack the Thai corpora

Two paths, set by `CONFIG["streaming"]`:

- **streaming=True** (default, **required for these two repos**): the SPAISS6F1
  datasets bundle many sub-corpora with *different* parquet schemas (`text` vs `txt`
  vs `body`, plus extra columns), which breaks `load_dataset`'s schema unification.
  So we enumerate the parquet files and read just the text column from each,
  normalizing to `{"text": ...}` — pulled from HF on the fly, nothing big hits disk.
  Train is an `IterableDataset` bounded by `max_steps`; eval is the first
  `eval_blocks` blocks.
- **streaming=False**: only works for *uniform-schema* datasets (downloads, caps to
  `max_train_docs`, normal split). It will CastError on these mixed repos — keep
  streaming on for them.""")
code("""
from itertools import chain
from datasets import load_dataset, concatenate_datasets, interleave_datasets, IterableDataset

# These repos are NOT uniform: they bundle many sub-corpora with DIFFERENT schemas
# (some have `text`, some `txt`, some `body`, plus extra columns). datasets' schema
# unification CastErrors on that. So we read each parquet file ourselves and keep
# only the text column (whatever it's called) -> uniform {"text": ...} stream.
TEXT_CANDIDATES = ["text", "txt", "body", "content", "document", "article", "raw_content"]

seq = CONFIG["sequence_length"]; eos = tok.eos_token or ""
def tok_fn(b): return tok([t+eos for t in b["text"]], add_special_tokens=False)
def group(b):
    ids = list(chain(*b["input_ids"]))
    total = (len(ids)//seq)*seq
    ch = [ids[i:i+seq] for i in range(0, total, seq)]
    return {"input_ids": ch, "labels": [c.copy() for c in ch]}

if CONFIG["streaming"]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()   # uses the token from the login() cell

    def gen_repo(repo_id):
        files = fs.glob(f"datasets/{repo_id}/**/*.parquet")
        print(f"  {repo_id}: {len(files)} parquet files", flush=True)
        for fp in files:
            try:
                with fs.open(fp) as fh:
                    pf = pq.ParquetFile(fh)
                    col = next((c for c in TEXT_CANDIDATES if c in pf.schema_arrow.names), None)
                    if col is None:
                        continue
                    for batch in pf.iter_batches(batch_size=1000, columns=[col]):
                        for v in batch.column(0).to_pylist():
                            if v and isinstance(v, str) and v.strip():
                                yield {"text": v}
            except Exception as e:
                print("  skip", fp.split("/")[-1], type(e).__name__, flush=True)

    streams = [IterableDataset.from_generator(gen_repo, gen_kwargs={"repo_id": ds_id})
               for ds_id, _split, _field in CONFIG["datasets"]]
    # all_exhausted -> use every doc from both corpora, not just the smaller one.
    stream = streams[0] if len(streams)==1 else interleave_datasets(
        streams, stopping_strategy="all_exhausted")
    stream = stream.shuffle(seed=CONFIG["seed"], buffer_size=CONFIG["shuffle_buffer"])
    toked = stream.map(tok_fn, batched=True, remove_columns=["text"])
    packed = toked.map(group, batched=True, remove_columns=["input_ids", "attention_mask"])
    eval_ds = packed.take(CONFIG["eval_blocks"])     # finite eval set
    train_ds = packed.skip(CONFIG["eval_blocks"])    # everything after -> training
    print(f"streaming ready | eval blocks: {CONFIG['eval_blocks']} | "
          f"train bounded by max_steps={CONFIG['max_steps']}")
else:
    def norm(ds, field):
        if field != "text" and field in ds.column_names:
            ds = ds.rename_column(field, "text")
        drop = [c for c in ds.column_names if c != "text"]
        return ds.remove_columns(drop) if drop else ds
    parts = []
    for ds_id, split, field in CONFIG["datasets"]:
        print("loading", ds_id)
        d = norm(load_dataset(ds_id, split=split), field)
        print("  ->", f"{len(d):,}", "docs"); parts.append(d)
    corpus = parts[0] if len(parts)==1 else concatenate_datasets(parts)
    if CONFIG["max_train_docs"]:
        corpus = corpus.select(range(min(CONFIG["max_train_docs"], len(corpus))))
    toked = corpus.map(tok_fn, batched=True, remove_columns=corpus.column_names, desc="tokenize")
    packed = toked.map(group, batched=True, remove_columns=toked.column_names, desc="pack")
    print(f"packed examples: {len(packed):,}  (~{len(packed)*seq/1e6:.1f}M tokens)")
    ds = packed.train_test_split(test_size=CONFIG["validation_fraction"], seed=CONFIG["seed"])
    train_ds, eval_ds = ds["train"], ds["test"]
    print("train", f"{len(train_ds):,}", "| val", f"{len(eval_ds):,}")
""")

# ---------------------------------------------------------------- (optional) drive
md("""## 7. (Optional) Persist checkpoints to Google Drive

Colab sessions are temporary. Mount Drive and point `output_dir` there to resume
training later. Skip this cell to keep checkpoints only for the session.""")
code("""
from google.colab import drive
drive.mount("/content/drive")
CONFIG["output_dir"] = "/content/drive/MyDrive/thai-1b-ckpt"
print("checkpoints ->", CONFIG["output_dir"])
""")

# ---------------------------------------------------------------- train
md("## 8. Pretrain (fp16 + 8-bit paged AdamW + gradient checkpointing)")
code("""
import os
from transformers import Trainer, TrainingArguments, default_data_collator
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model.config.use_cache = False  # required with gradient checkpointing
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# 8-bit optimizer keeps Adam state ~1 byte/param so 1B fits a 16GB T4.
try:
    import bitsandbytes  # noqa: F401
    optim = "paged_adamw_8bit"
except Exception:
    optim = "adamw_torch"
    print("bitsandbytes unavailable -> falling back to", optim, "(may OOM at 1B)")
print("optimizer:", optim, "| dtype:", "bf16" if BF16_OK else "fp16")

args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
    warmup_ratio=CONFIG["warmup_ratio"],
    max_steps=CONFIG["max_steps"],
    lr_scheduler_type="cosine",
    bf16=BF16_OK, fp16=not BF16_OK,
    optim=optim,
    gradient_checkpointing=True,
    logging_steps=CONFIG["logging_steps"],
    save_steps=CONFIG["save_steps"],
    eval_strategy="steps", eval_steps=CONFIG["eval_steps"],
    save_total_limit=2, seed=CONFIG["seed"], report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=eval_ds, data_collator=default_data_collator,
                  processing_class=tok)

resume = os.path.isdir(CONFIG["output_dir"]) and any(
    p.startswith("checkpoint-") for p in os.listdir(CONFIG["output_dir"]))
trainer.train(resume_from_checkpoint=resume)
trainer.save_model(CONFIG["output_dir"] + "/final")
tok.save_pretrained(CONFIG["output_dir"] + "/final")
print("saved ->", CONFIG["output_dir"] + "/final")
""")

# ---------------------------------------------------------------- eval
md("## 9. Sanity check — perplexity + Thai generation")
code("""
import math, torch
m = trainer.evaluate()
if "eval_loss" in m:
    print(f"eval_loss {m['eval_loss']:.4f} | perplexity {math.exp(m['eval_loss']):.1f}")

model.config.use_cache = True
model.eval()
for prompt in ["ประเทศไทยมีเมืองหลวงคือ", "ปัญญาประดิษฐ์ (AI) คือ"]:
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=48, do_sample=True,
                             temperature=0.8, top_p=0.9,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    print("\\nPROMPT:", prompt, "\\nOUTPUT:", tok.decode(out[0], skip_special_tokens=True))
""")

# ---------------------------------------------------------------- notes
md("""## Notes & scaling

- **Data is pulled straight from HF.** With `streaming=True` (default) shards are
  fetched on the fly and never fully written to Colab disk — so corpus size is not
  bounded by the ~78 GB disk, only `max_steps` bounds how much you consume. With
  `streaming=False`, the full splits download first, then `max_train_docs` caps them.
- **Streaming + resume:** `resume_from_checkpoint` restores model/optimizer, but the
  stream restarts from the beginning (no exact data-position checkpoint). Fine for
  continued pretraining; just know later steps may re-see early shards.
- **OOM at 1B?** Lower `CONFIG["target_params"]` (e.g. 700M) or `hidden_size`, keep
  batch size 1 and 8-bit optim. The free T4 has 16 GB VRAM.
- **A real pretrain** needs far more than 2000 steps and tens of B of tokens — raise
  `max_steps`, keep streaming on, and checkpoint to Drive so you can resume across
  Colab sessions (re-run cells 1-7 then cell 8; it auto-resumes).
- **Throughput:** a T4 (~25 TFLOPS sustained) is fine for a dev/demo run; a full
  ~20B-token 1B pretrain is weeks on one T4 — use a multi-GPU box for the real thing.
- This mirrors the local pipeline in `../` (reinit path). `structured` (warm-start
  prune of the full base) needs ~5B params loaded — too big for a T4; use reinit here.
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 0,
}
out = Path(__file__).parent / "thai_1b_t4.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out, f"({len(cells)} cells)")
