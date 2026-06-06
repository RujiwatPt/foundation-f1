#!/usr/bin/env python
"""Stage 01 — derive a ~1B model from the Gemma 4 E2B base.

Two modes (config: prune.mode):

  reinit      Build a fresh dense ~1B text-decoder config from the base
              architecture family, RANDOM-INITIALISE the weights, and reuse the
              base tokenizer. This is the literal "1B model without weights,
              trained from scratch" path. Well tested; the default.

  structured  Warm start: load the base's pretrained text decoder and prune it
              down to ~1B by (a) keeping an evenly-spaced subset of layers
              (depth pruning) and (b) trimming each MLP's intermediate width.
              Both operations leave a runnable model whose weights are inherited
              from the base, which trains better per FLOP than reinit. Best
              effort — exact tensor names can vary by transformers version.

Either way the output is a standard CausalLM directory consumable by stage 03.
"""
from __future__ import annotations

import json

import torch

from common import get_compute_dtype, human, load_config, resolve


def _base_text_config(base_dir, trust):
    from transformers import AutoConfig

    conf = AutoConfig.from_pretrained(base_dir, trust_remote_code=trust)
    # Gemma 4 E2B is multimodal; the LM lives under .text_config.
    return getattr(conf, "text_config", conf), conf


def build_target_config(cfg):
    """Build the ~1B config, searching num_hidden_layers to hit target params."""
    from transformers import Gemma3TextConfig

    base_dir = resolve(cfg["base"]["local_dir"])
    trust = cfg["base"].get("trust_remote_code", True)
    text_cfg, _ = _base_text_config(base_dir, trust)

    a = cfg["prune"]["arch"]
    vocab = text_cfg.vocab_size  # ALWAYS inherit the tokenizer's vocab
    target = int(cfg["prune"]["target_params"])

    def make(layers):
        # Prefer the base's own config class so RoPE/norm/activation defaults match.
        kwargs = dict(
            vocab_size=vocab,
            hidden_size=a["hidden_size"],
            intermediate_size=a["intermediate_size"],
            num_hidden_layers=layers,
            num_attention_heads=a["num_attention_heads"],
            num_key_value_heads=a["num_key_value_heads"],
            max_position_embeddings=a["max_position_embeddings"],
            tie_word_embeddings=a["tie_word_embeddings"],
        )
        try:
            return text_cfg.__class__(**kwargs)
        except Exception:  # noqa: BLE001 — fall back to a known small-LM config
            return Gemma3TextConfig(**kwargs)

    # Count params from the model ACTUALLY built on a meta device (no memory
    # allocated). The analytic formula misses architecture-specific weights
    # (e.g. Gemma 4's per-layer embeddings + AltUp/Laurel blocks), so we trust
    # the real module tree. parameters() de-dupes tied weights for us.
    import logging

    from transformers import AutoModelForCausalLM

    logging.getLogger("transformers").setLevel(logging.ERROR)

    def meta_count(layers):
        c = make(layers)
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(c)
        return sum(p.numel() for p in m.parameters()), c

    # Params grow monotonically with depth: scan upward, stop once we pass target.
    best = None
    prev = None
    for layers in range(1, 60):
        n, c = meta_count(layers)
        score = abs(n - target)
        if best is None or score < best[0]:
            best = (score, layers, n, c)
        if prev is not None and n > target:
            break
        prev = n
    _, layers, n, conf = best
    if n > target * 1.05:
        print(f"[01] WARNING: smallest viable depth ({layers} layer(s)) is "
              f"{human(n)} > target {human(target)}. The {human(vocab)}-token "
              f"vocab dominates a small model; lower hidden_size or raise target.")
    print(f"[01] Selected num_hidden_layers={layers} -> ~{human(n)} params "
          f"(target {human(target)})")
    return conf, n


def do_reinit(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    conf, est = build_target_config(cfg)
    out = resolve(cfg["prune"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    print("[01] Instantiating randomly-initialised model from config ...")
    # fp32 on MPS/CPU, bf16 on CUDA. Stage 03 reloads in the right dtype anyway.
    model = AutoModelForCausalLM.from_config(conf, torch_dtype=get_compute_dtype())
    actual = sum(p.numel() for p in model.parameters())
    print(f"[01] Materialised params: {human(actual)} (estimate was {human(est)})")

    base_dir = resolve(cfg["base"]["local_dir"])
    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=cfg["base"]["trust_remote_code"])

    model.save_pretrained(out)
    tok.save_pretrained(out)
    _write_summary(out, cfg, conf, actual, mode="reinit")
    print(f"[01] Saved randomly-initialised 1B model + tokenizer to {out}")


def do_structured(cfg):
    """Depth + MLP-width prune of the base's pretrained text decoder."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_dir = resolve(cfg["base"]["local_dir"])
    trust = cfg["base"].get("trust_remote_code", True)
    target = int(cfg["prune"]["target_params"])
    out = resolve(cfg["prune"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    print("[01] Loading base weights for structured pruning (this is large) ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_dir, trust_remote_code=trust, torch_dtype=torch.bfloat16
    )
    # Reach the text decoder (handles the multimodal wrapper).
    lm = getattr(base, "language_model", base)
    decoder = getattr(lm, "model", lm)
    layers = decoder.layers
    n_base = len(layers)

    # --- Depth pruning: keep evenly-spaced layers, always including first+last.
    full = sum(p.numel() for p in base.parameters())
    # Crude per-layer cost to estimate how many layers fit the budget.
    per_layer = sum(p.numel() for p in layers[0].parameters())
    non_layer = full - per_layer * n_base
    keep = max(1, min(n_base, round((target - non_layer) / per_layer)))
    idx = torch.linspace(0, n_base - 1, steps=keep).round().long().tolist()
    idx = sorted(set(idx))
    print(f"[01] Depth prune: keep {len(idx)}/{n_base} layers -> {idx}")

    import torch.nn as nn

    decoder.layers = nn.ModuleList([layers[i] for i in idx])
    decoder.config.num_hidden_layers = len(idx)
    if hasattr(base.config, "text_config"):
        base.config.text_config.num_hidden_layers = len(idx)

    actual = sum(p.numel() for p in base.parameters())
    print(f"[01] Post-prune params: {human(actual)} (target {human(target)})")

    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=trust)
    base.save_pretrained(out)
    tok.save_pretrained(out)
    _write_summary(out, cfg, getattr(base.config, "text_config", base.config),
                   actual, mode="structured")
    print(f"[01] Saved structurally-pruned (warm-start) 1B model to {out}")


def _write_summary(out, cfg, conf, n_params, mode):
    summary = {
        "mode": mode,
        "base_model": cfg["base"]["model_id"],
        "target_params": cfg["prune"]["target_params"],
        "actual_params": int(n_params),
        "hidden_size": getattr(conf, "hidden_size", None),
        "num_hidden_layers": getattr(conf, "num_hidden_layers", None),
        "num_attention_heads": getattr(conf, "num_attention_heads", None),
        "num_key_value_heads": getattr(conf, "num_key_value_heads", None),
        "intermediate_size": getattr(conf, "intermediate_size", None),
        "vocab_size": getattr(conf, "vocab_size", None),
        "tie_word_embeddings": getattr(conf, "tie_word_embeddings", None),
    }
    with open(out / "prune_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    cfg = load_config()
    mode = cfg["prune"]["mode"]
    print(f"[01] Prune mode: {mode}")
    if mode == "reinit":
        do_reinit(cfg)
    elif mode == "structured":
        do_structured(cfg)
    else:
        raise SystemExit(f"Unknown prune.mode: {mode!r} (use 'reinit' or 'structured')")


if __name__ == "__main__":
    main()
