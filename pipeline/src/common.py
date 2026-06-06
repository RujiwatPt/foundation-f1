"""Shared helpers for the foundation-F1 pipeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Repo-relative anchor: pipeline/ is the parent of src/.
PIPELINE_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the pipeline config.

    Resolution order: explicit `path` arg > env var F1_CONFIG > pipeline/config.yaml.
    The env var lets `make CONFIG=config.m2.yaml ...` switch every stage at once.
    """
    if path is None:
        path = os.environ.get("F1_CONFIG")
    cfg_path = resolve(path) if path else PIPELINE_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> str:
    """Pick the best available backend: CUDA > Apple MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_compute_dtype(device: str | None = None):
    """Training/inference dtype per backend.

    CUDA -> bfloat16. MPS/CPU -> float32 (bf16/fp16 training on MPS is still
    unreliable in PyTorch; fp32 is the safe, correct choice on Apple Silicon).
    """
    import torch

    device = device or get_device()
    return torch.bfloat16 if device == "cuda" else torch.float32


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a config path relative to the pipeline root unless absolute."""
    p = Path(path)
    return p if p.is_absolute() else (PIPELINE_ROOT / p).resolve()


def human(n: int | float) -> str:
    """Human-readable parameter count, e.g. 1.02B / 734.1M."""
    n = float(n)
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"


def count_params_from_config(cfg) -> int:
    """Estimate total parameter count of a decoder-only LM from its config.

    Covers token embeddings, per-layer attention (with GQA) + gated MLP, and the
    LM head (only when untied). Accurate to a few percent — enough for the
    architecture search in 01_prune_to_1b.
    """
    V = cfg.vocab_size
    H = cfg.hidden_size
    L = cfg.num_hidden_layers
    I = cfg.intermediate_size
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads) or n_heads
    head_dim = getattr(cfg, "head_dim", None) or (H // n_heads)

    embeddings = V * H
    q = H * (n_heads * head_dim)
    kv = 2 * H * (n_kv * head_dim)
    o = (n_heads * head_dim) * H
    attn = q + kv + o
    mlp = 3 * H * I              # gated MLP: gate + up + down
    norms = 2 * H               # input + post-attention RMSNorm per layer
    per_layer = attn + mlp + norms

    total = embeddings + L * per_layer + H  # + final norm
    if not getattr(cfg, "tie_word_embeddings", True):
        total += V * H          # separate LM head
    return int(total)
