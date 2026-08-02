# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Codexa v1 builds a decoder-only Transformer base model from scratch with a
custom model, tokenizer, and training pipeline on one RTX 4080 (16 GB VRAM).
The only active training target is the 921,773,568-parameter configuration in
`configs/1b.yaml`. Follow `documentation/planning/PHASE_PLAN.md`; do not restore
superseded small-model, chat, SFT, hosted-notebook, or old-run workflows.

## Commands

Run the full suite with pytest. Tests also remain executable as plain Python
modules from the repository root:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m tests.test_model
.venv/bin/python -m tests.test_config
.venv/bin/python -m tests.test_data_cleaning
.venv/bin/python -m tests.test_data_pipeline
.venv/bin/python -m tests.test_tokenizer
.venv/bin/python -m tests.test_token_data
```

Use pytest discovery for normal validation and individual modules for focused
checks.

Pipeline scripts (all support `--help`, all require running from the repo root so `src` imports resolve, or use the `sys.path` shim already in each script):

```bash
.venv/bin/python scripts/prepare_dataset.py <input files...> --output-dir data/processed --dataset-name <name> --license <license>
.venv/bin/python scripts/train_tokenizer.py <cleaned jsonl...> --output-dir <dir> --vocab-size 8192
.venv/bin/python scripts/tokenize_dataset.py --train-jsonl data/processed/train.jsonl --validation-jsonl data/processed/validation.jsonl --tokenizer <tokenizer.json> --output-dir data/tokenized --model-vocab-size 8192 --context-length 256
.venv/bin/python scripts/inspect_tokenizer.py ...
.venv/bin/python scripts/inspect_token_data.py <manifest.json>
```

No linter/formatter is configured in this repo — don't assume `ruff`/`black`/`flake8` are available.

## Architecture

**Pipeline order:** raw text → `scripts/prepare_dataset.py` (clean + dedupe + split) → `scripts/train_tokenizer.py` (BPE) → `scripts/tokenize_dataset.py` (binary token arrays) → `scripts/train.py` → checkpoints → generation and evaluation.

- **`src/model.py`** — the decoder-only Transformer itself: `ModelConfig` (validates dimensions in `__post_init__`), RMSNorm, causal self-attention (via `F.scaled_dot_product_attention`, always causal), SwiGLU `FeedForward`, pre-norm `TransformerBlock`, and `LanguageModel`. Optional embedding/lm_head weight tying is handled by pointing `lm_head.weight` at `token_embeddings.weight` after init — `count_parameters` dedupes tied parameters by `id()` so tied models aren't double-counted. `LanguageModel.forward` validates input dtype/shape/context length itself and returns `(logits, loss_or_None)`.

- **`src/config.py`** — typed YAML config loading (`ProjectConfig` = `ModelConfig` + `TrainingConfig`). `load_config()` strictly validates YAML: unknown or missing keys in either the `model:` or `training:` section raise immediately (see `_validate_keys`), so config files must exactly match the dataclass fields in `src/model.py`/`src/config.py`. The active full configuration is `configs/1b.yaml`; `configs/smoke.yaml` and `configs/tiny_overfit.yaml` are deterministic test fixtures.

- **`src/tokenizer.py`** — byte-level BPE via HuggingFace `tokenizers`, trained from scratch (no pretrained tokenizer). Special tokens are pinned to fixed IDs by construction order: `<pad>=0, <bos>=1, <eos>=2, <unk>=3` (`SPECIAL_TOKENS` in that order) — `validate_tokenizer()` enforces this on every load, and downstream code (`src/token_data.py`) hardcodes the assumption that `<eos>` is ID 2. Training writes both `tokenizer.json` and a `tokenizer_manifest.json` recording SHA-256 checksums of inputs/output and tokenizer settings for reproducibility.

- **`src/data/`** — dataset prep as small composable stages: `io.py` (read/write `.txt`/`.jsonl`/`.ndjson`, the shared `TextDocument` type), `cleaning.py` (Unicode NFKC normalization + exact-duplicate removal by content hash), `split.py` (deterministic train/validation split — assignment is a SHA-256 hash of `seed + document_id_or_text`, not random, so splits are stable across runs and don't depend on document order), `statistics.py` (dataset stat aggregation for manifests).

- **`src/token_data.py`** — turns cleaned JSONL into memory-mappable binary token arrays (`train.bin`/`validation.bin`, dtype auto-chosen as `uint16` or `uint32` from vocab size via `choose_token_dtype`). Each document's tokens are followed by an EOS token; a JSON index (`train_index.json`/`validation_index.json`) records per-document byte/token offsets for verification. `MemmapTokenDataset` reads fixed-length `(context_length + 1)`-token windows via `np.memmap` and returns `(input_ids, labels)` shifted by one position — this is where next-token-prediction framing happens, not in the model. `build_token_data` writes to temp files and atomically renames into place (`os.replace`) so a crash mid-build can't leave partial/corrupt outputs; `inspect_token_data` independently re-verifies checksums, index contiguity, and EOS placement against the manifest.

- **Manifests everywhere:** every pipeline stage (dataset prep, tokenizer training, token-data build) writes a JSON manifest recording exact inputs, SHA-256 checksums, and settings used, so any artifact can be traced back to how it was produced. When adding a new pipeline stage, follow this convention.

- **Validation style:** dataclasses validate their own fields in `__post_init__` (see `ModelConfig`, `TrainingConfig`) rather than relying on callers to check first; functions raise `ValueError`/`TypeError` with specific messages naming the bad field and value rather than asserting silently. Match this style in new code.

## Environment

- Python 3.14, `.venv` already created with PyTorch (CUDA 13.0 build), NumPy, PyYAML, and `tokenizers` installed. Dependencies are recorded in `requirements.txt` and `requirements-dev.txt`.
- Target hardware is a single RTX 4080 (16 GB VRAM); training-related choices (precision, batch size, gradient accumulation) in configs are made with that budget in mind.
