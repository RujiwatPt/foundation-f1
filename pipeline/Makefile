# Foundation-F1 — Thai 1B foundation model pipeline
# Usage: make <target>   (run from pipeline/)

PY ?= python
SRC = src

# Select a config for every stage at once:  make CONFIG=config.m2.yaml all
CONFIG ?= config.yaml
export F1_CONFIG = $(CONFIG)
# Let unsupported MPS ops fall back to CPU instead of erroring (Apple Silicon).
export PYTORCH_ENABLE_MPS_FALLBACK = 1
# Avoid tokenizers fork/parallelism warnings under datasets multiprocessing.
export TOKENIZERS_PARALLELISM = false

.PHONY: all base prune data train eval clean m2 smoke help

help:
	@echo "Targets (override config with CONFIG=config.m2.yaml):"
	@echo "  make base   - 00: download base tokenizer + config (needs HF_TOKEN)"
	@echo "  make prune  - 01: derive the model (reinit | structured)"
	@echo "  make data   - 02: build the tokenized/packed Thai dataset"
	@echo "  make train  - 03: pretrain"
	@echo "  make eval   - 04: perplexity + sample generations"
	@echo "  make all    - run 00 -> 04 in order"
	@echo "  make m2     - run 00 -> 04 with the Apple Silicon profile"
	@echo "  current CONFIG = $(CONFIG)"

# Convenience: full run with the M2 Pro profile.
m2:
	$(MAKE) CONFIG=config.m2.yaml all

# Self-contained smoke test: no gated download, no HF auth. Fabricates a tiny
# stand-in base, then runs prune -> data -> train -> eval on local parquet.
smoke:
	F1_CONFIG=config.test.yaml $(PY) $(SRC)/_smoke_setup.py
	F1_CONFIG=config.test.yaml $(PY) $(SRC)/01_prune_to_1b.py
	F1_CONFIG=config.test.yaml $(PY) $(SRC)/02_prepare_data.py
	F1_CONFIG=config.test.yaml $(PY) $(SRC)/03_pretrain.py
	F1_CONFIG=config.test.yaml $(PY) $(SRC)/04_eval.py
	@echo "[smoke] OK — all stages ran. Clean up with: make clean-smoke"

clean-smoke:
	rm -rf artifacts/base-test artifacts/model-test-init \
	       artifacts/tokenized-test artifacts/checkpoints-test

all: base prune data train eval

base:
	$(PY) $(SRC)/00_download_base.py

prune:
	$(PY) $(SRC)/01_prune_to_1b.py

data:
	$(PY) $(SRC)/02_prepare_data.py

train:
	$(PY) $(SRC)/03_pretrain.py

eval:
	$(PY) $(SRC)/04_eval.py

# Multi-GPU training: make train-dist N=8
train-dist:
	torchrun --nproc_per_node=$(N) $(SRC)/03_pretrain.py

clean:
	rm -rf artifacts/tokenized artifacts/checkpoints
