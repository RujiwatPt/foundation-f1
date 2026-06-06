#!/usr/bin/env python
"""Fabricate a tiny stand-in base so the pipeline can be smoke-tested with no
gated download and no HF auth. Creates artifacts/base-test (small open tokenizer
+ a tiny Gemma text config). Used by `make smoke`.
"""
from pathlib import Path

from transformers import AutoTokenizer, Gemma3TextConfig

from common import resolve


def main():
    base = resolve("artifacts/base-test")
    base.mkdir(parents=True, exist_ok=True)
    # Tiny open tokenizer stands in for the gated Gemma tokenizer.
    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(base)
    cfg = Gemma3TextConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, tie_word_embeddings=True,
    )
    cfg.save_pretrained(base)
    print(f"[smoke] stand-in base ready at {base} (vocab {len(tok)})")


if __name__ == "__main__":
    main()
