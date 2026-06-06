#!/usr/bin/env python
"""Stage 03 — pretrain the 1B model on the packed Thai corpus.

Standard causal-LM next-token pretraining with the HF Trainer:
  * bf16, gradient checkpointing, cosine LR with warmup
  * data already packed to fixed length in stage 02 (labels == input_ids)
  * checkpoints + resume, periodic perplexity on the val split

Single GPU:   python 03_pretrain.py
Multi GPU:    torchrun --nproc_per_node=N 03_pretrain.py
"""
from __future__ import annotations

import math
import os

from common import get_device, human, load_config, resolve


def main():
    cfg = load_config()
    t = cfg["train"]

    from datasets import load_from_disk
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )

    model_dir = resolve(cfg["prune"]["output_dir"])
    data_dir = resolve(cfg["data"]["tokenized_dir"])
    out = resolve(t["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    print(f"[03] Loading initial 1B model from {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.config.use_cache = False  # required with gradient checkpointing
    print(f"[03] Trainable params: {human(sum(p.numel() for p in model.parameters()))}")

    device = get_device()
    use_bf16 = device == "cuda"   # bf16/fp16 training is unreliable on MPS -> fp32
    print(f"[03] Backend: {device} | bf16={use_bf16}")

    print(f"[03] Loading tokenized dataset from {data_dir}")
    ds = load_from_disk(str(data_dir))
    if hasattr(ds, "keys"):  # DatasetDict
        train_ds = ds["train"]
        eval_ds = ds.get("validation")
    else:
        train_ds, eval_ds = ds, None

    # min_lr via cosine: HF exposes it through lr_scheduler_kwargs on recent versions.
    min_lr_ratio = float(t.get("min_lr_ratio", 0.1))

    args_kwargs = dict(
        output_dir=str(out),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        adam_beta1=float(t["adam_beta1"]),
        adam_beta2=float(t["adam_beta2"]),
        adam_epsilon=float(t["adam_epsilon"]),
        max_grad_norm=float(t["max_grad_norm"]),
        warmup_ratio=float(t["warmup_ratio"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        bf16=use_bf16,
        gradient_checkpointing=bool(t["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        dataloader_num_workers=t["dataloader_num_workers"],
        dataloader_pin_memory=(device == "cuda"),  # MPS doesn't support pinning
        seed=t["seed"],
        report_to=t["report_to"],
        run_name=t["run_name"],
    )
    if t.get("max_steps"):
        args_kwargs["max_steps"] = int(t["max_steps"])
    if t.get("num_train_epochs"):
        args_kwargs["num_train_epochs"] = float(t["num_train_epochs"])
    if eval_ds is not None:
        args_kwargs.update(eval_strategy="steps", eval_steps=t["eval_steps"])

    # Cosine-with-floor where supported; harmless if the kwarg is ignored.
    try:
        args = TrainingArguments(
            **args_kwargs,
            lr_scheduler_kwargs={"min_lr_rate": min_lr_ratio},
        )
    except TypeError:
        args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        processing_class=tok,
    )

    resume = bool(t.get("resume", True)) and any(
        p.name.startswith("checkpoint-") for p in out.glob("checkpoint-*")
    )
    print(f"[03] Starting training (resume={resume}) ...")
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(str(out / "final"))
    tok.save_pretrained(str(out / "final"))

    if eval_ds is not None:
        metrics = trainer.evaluate()
        loss = metrics.get("eval_loss")
        if loss is not None:
            print(f"[03] Final eval loss {loss:.4f} | perplexity {math.exp(loss):.2f}")
    print(f"[03] Done. Final model at {out / 'final'}")


if __name__ == "__main__":
    main()
