#!/usr/bin/env python
"""Stage 02 — build the tokenized pretraining dataset.

Sources (config: data):
  * hf_datasets   — the two SPAISS6F1 Thai corpora (and any others you add)
  * local_parquet — parquet already in this repo (schema: text, source)

Pipeline: load -> normalise to a single `text` column -> concatenate all
sources -> tokenize -> pack into fixed-length blocks of `sequence_length`
tokens (each token also serves as the next-token label) -> save to disk.

Packing (concatenate-then-chunk) is the standard pretraining layout: no padding
waste, every position contributes a training signal.
"""
from __future__ import annotations

from itertools import chain

from common import human, load_config, resolve


def load_sources(cfg):
    from datasets import Dataset, concatenate_datasets, load_dataset

    parts = []

    # --- Hugging Face hub datasets ---
    for spec in cfg["data"].get("hf_datasets", []) or []:
        ds_id = spec["id"]
        split = spec.get("split", "train")
        field = spec.get("text_field", cfg["data"]["text_field"])
        print(f"[02] Loading HF dataset {ds_id} [{split}] (text field: {field})")
        ds = load_dataset(ds_id, split=split)
        ds = _normalise(ds, field)
        print(f"[02]   -> {len(ds):,} docs")
        parts.append(ds)

    # --- Local parquet ---
    import glob

    patterns = cfg["data"].get("local_parquet", []) or []
    files = sorted({f for pat in patterns for f in glob.glob(str(resolve(pat)))})
    if files:
        field = cfg["data"]["text_field"]
        print(f"[02] Loading {len(files)} local parquet file(s)")
        ds = load_dataset("parquet", data_files=files, split="train")
        ds = _normalise(ds, field)
        print(f"[02]   -> {len(ds):,} docs")
        parts.append(ds)

    if not parts:
        raise SystemExit("[02] No data sources resolved — check config.data.")

    combined = parts[0] if len(parts) == 1 else concatenate_datasets(parts)

    cap = cfg["data"].get("max_train_docs")
    if cap:
        combined = combined.select(range(min(cap, len(combined))))
        print(f"[02] Capped to {len(combined):,} docs (max_train_docs)")

    print(f"[02] Combined corpus: {len(combined):,} docs")
    return combined


def _normalise(ds, field):
    """Reduce any dataset to a single `text` column."""
    if field != "text" and field in ds.column_names:
        ds = ds.rename_column(field, "text")
    if "text" not in ds.column_names:
        raise SystemExit(f"[02] Expected a '{field}'/'text' column, got {ds.column_names}")
    drop = [c for c in ds.column_names if c != "text"]
    return ds.remove_columns(drop) if drop else ds


def main():
    cfg = load_config()
    from transformers import AutoTokenizer

    model_dir = resolve(cfg["prune"]["output_dir"])
    tok = AutoTokenizer.from_pretrained(model_dir)
    seq_len = int(cfg["data"]["sequence_length"])
    num_proc = int(cfg["data"]["num_proc"])
    seed = int(cfg["data"]["seed"])
    out = resolve(cfg["data"]["tokenized_dir"])
    out.mkdir(parents=True, exist_ok=True)

    ds = load_sources(cfg)

    eos = tok.eos_token or ""

    def tok_fn(batch):
        # Append EOS so the model learns document boundaries.
        return tok([t + eos for t in batch["text"]], add_special_tokens=False)

    print("[02] Tokenizing ...")
    tokenized = ds.map(
        tok_fn, batched=True, num_proc=num_proc,
        remove_columns=ds.column_names, desc="tokenize",
    )

    def group(batch):
        ids = list(chain(*batch["input_ids"]))
        total = (len(ids) // seq_len) * seq_len
        chunks = [ids[i:i + seq_len] for i in range(0, total, seq_len)]
        return {"input_ids": chunks, "labels": [c.copy() for c in chunks]}

    print(f"[02] Packing into blocks of {seq_len} tokens ...")
    # remove_columns is required: group() changes the row count, so the leftover
    # attention_mask column would mismatch the new input_ids length and error.
    packed = tokenized.map(
        group, batched=True, num_proc=num_proc,
        remove_columns=tokenized.column_names, desc="pack",
    )

    n_tokens = len(packed) * seq_len
    print(f"[02] Packed examples: {len(packed):,}  (~{human(n_tokens)} tokens)")

    val_frac = float(cfg["data"]["validation_fraction"])
    if val_frac > 0 and len(packed) > 1:
        split = packed.train_test_split(test_size=val_frac, seed=seed)
        split["validation"] = split.pop("test")
        split.save_to_disk(str(out))
        print(f"[02] Saved train={len(split['train']):,} "
              f"val={len(split['validation']):,} to {out}")
    else:
        packed.save_to_disk(str(out))
        print(f"[02] Saved {len(packed):,} examples to {out}")


if __name__ == "__main__":
    main()
