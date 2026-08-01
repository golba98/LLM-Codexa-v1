# Phase 20A — 1B Architecture

## Goal

Define an approximately 1B-parameter decoder-only Transformer that fits the
RTX 4080's 16 GB VRAM limit using BF16 and gradient accumulation.

## Work

1. Choose layers, hidden width, attention heads, context length, vocabulary,
   and tied or untied embeddings.
2. Add a dedicated 1B YAML configuration; do not alter the frozen 250M config.
3. Verify the exact parameter count with `src/model.py`.
4. Run a BF16 forward/backward benchmark and record VRAM use.
5. Reduce micro-batch size or add memory-saving techniques if required.

## Acceptance criteria

- Parameter count is recorded exactly.
- A short CUDA BF16 run completes without OOM, NaN, or Inf values.
- Configuration and benchmark output are saved before pretraining.
