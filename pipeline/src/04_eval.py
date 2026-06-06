#!/usr/bin/env python
"""Stage 04 — quick sanity eval of the pretrained 1B model.

  * Perplexity on the held-out validation split (if present).
  * A few free-form Thai generations to eyeball coherence.

This is a smoke test, not the real evaluation. The full benchmark plan lives in
foundation-bundle/evalplan.md (golden set, retrieval/answer/adversarial axes).
"""
from __future__ import annotations

import math

import torch

from common import get_compute_dtype, get_device, load_config, resolve


def main():
    cfg = load_config()
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
    from torch.utils.data import DataLoader

    final = resolve(cfg["train"]["output_dir"]) / "final"
    model_dir = final if final.exists() else resolve(cfg["prune"]["output_dir"])
    print(f"[04] Loading model from {model_dir}")

    tok = AutoTokenizer.from_pretrained(model_dir)
    device = get_device()
    print(f"[04] Backend: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=get_compute_dtype(device)
    ).to(device).eval()

    # --- Perplexity on validation ---
    data_dir = resolve(cfg["data"]["tokenized_dir"])
    if data_dir.exists():
        ds = load_from_disk(str(data_dir))
        val = ds.get("validation") if hasattr(ds, "keys") else None
        if val is not None and len(val) > 0:
            val = val.select(range(min(256, len(val))))
            dl = DataLoader(val, batch_size=4, collate_fn=default_data_collator)
            losses, n = 0.0, 0
            with torch.no_grad():
                for batch in dl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = model(**batch)
                    losses += out.loss.item()
                    n += 1
            mean = losses / max(n, 1)
            print(f"[04] Val loss {mean:.4f} | perplexity {math.exp(mean):.2f} "
                  f"(over {len(val)} packed blocks)")

    # --- Generations ---
    print("\n[04] Sample generations:")
    for prompt in cfg["eval"]["prompts"]:
        ids = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **ids, max_new_tokens=cfg["eval"]["max_new_tokens"],
                do_sample=True, temperature=0.8, top_p=0.9,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        text = tok.decode(gen[0], skip_special_tokens=True)
        print(f"\n  PROMPT: {prompt}\n  OUTPUT: {text}")


if __name__ == "__main__":
    main()
