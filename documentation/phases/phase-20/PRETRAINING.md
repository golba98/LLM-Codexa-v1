# Phase 20B — 1B From-Scratch Pretraining

## Goal

Train the new 1B model from randomly initialized weights using a pinned,
licensed Hugging Face text mixture. Hugging Face supplies data only; the
Codexa tokenizer, architecture, weights, and training run remain ours.

## Data policy

- Prefer broad text such as FineWeb-Edu or another clearly licensed source.
- Pin revisions and record licenses, source files, sizes, and SHA-256 checksums.
- Prepare a 1B–2B-token subset instead of storing unused duplicate data.
- Validate cleaning, deduplication, token ranges, EOS placement, and splits.

## Run policy

- Freeze model, tokenizer, data, and YAML configuration before the full run.
- Keep best/latest checkpoints and only necessary milestones.
- Monitor loss, validation, throughput, temperature, VRAM, disk, and recovery.
- Estimated duration on the RTX 4080: approximately 1–3 days.

## Acceptance criteria

- The requested token target is reached.
- The best checkpoint passes checksum and reload validation.
- A base-model generation smoke test is recorded.
