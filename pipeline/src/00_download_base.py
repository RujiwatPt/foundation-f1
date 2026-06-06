#!/usr/bin/env python
"""Stage 00 — download the base model (Gemma 4 E2B).

We need two things from the base:
  1. the tokenizer  (reused verbatim by the 1B model)
  2. the model config / architecture (template for the derived 1B model)

Gemma is gated on the Hub: accept the license once at
https://huggingface.co/google/gemma-4-E2B-it and export HF_TOKEN (or run
`huggingface-cli login`) before running this.
"""
from __future__ import annotations

import sys

from common import human, load_config, resolve


def main() -> None:
    cfg = load_config()
    model_id = cfg["base"]["model_id"]
    local_dir = resolve(cfg["base"]["local_dir"])
    trust = cfg["base"].get("trust_remote_code", True)
    local_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig, AutoTokenizer

    print(f"[00] Downloading tokenizer + config for {model_id} ...")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
        conf = AutoConfig.from_pretrained(model_id, trust_remote_code=trust)
    except Exception as e:  # noqa: BLE001
        print(
            "\n[00] FAILED to fetch the base model. Most common causes:\n"
            "  * License not accepted — visit the model page and click 'Agree'.\n"
            "  * No auth token — run `huggingface-cli login` or export HF_TOKEN.\n"
            f"\nUnderlying error:\n  {e}",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    tok.save_pretrained(local_dir)
    conf.save_pretrained(local_dir)

    # Gemma 4 E2B is multimodal/MatFormer; the text-config may be nested.
    text_cfg = getattr(conf, "text_config", conf)
    vocab = getattr(text_cfg, "vocab_size", None) or getattr(conf, "vocab_size", None)

    print(f"[00] Saved tokenizer + config to {local_dir}")
    print(f"[00] Tokenizer vocab size : {len(tok)}")
    print(f"[00] Config vocab size    : {vocab}")
    print(f"[00] Base hidden_size     : {getattr(text_cfg, 'hidden_size', '?')}")
    print(f"[00] Base num_layers      : {getattr(text_cfg, 'num_hidden_layers', '?')}")
    print(f"[00] Base model_type      : {getattr(conf, 'model_type', '?')}")
    print(f"[00] Done. (tokenizer len {human(len(tok))})")


if __name__ == "__main__":
    main()
