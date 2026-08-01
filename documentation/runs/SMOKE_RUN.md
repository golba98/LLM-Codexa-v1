# Phase 11 Smoke Run

## Scope

The smoke run used a deterministic synthetic expansion of the repository's
original sample fixture. It is pipeline validation data, not a representative
pretraining corpus. `scripts/build_smoke_dataset.py` reproduces the cleaned and
tokenized splits without downloading external data.

## Dataset

- Raw documents: 10,000
- Training documents: 9,472
- Validation documents: 528
- Training tokens stored: 1,354,519
- Validation tokens stored: 75,481
- Context length: 256
- Host DataLoader benchmark: 42.7M tokens/second

## Training result

Command settings: smoke configuration, CUDA BF16, micro-batch 8, gradient
accumulation 4, 125 optimizer updates, and four validation batches at step 100.

- Tokens processed: 1,024,000
- Initial training loss: 9.069759
- Final training loss: 1.143102
- Step-100 validation loss: 2.628187
- Median steady training throughput: 236,482 tokens/second
- Peak allocated VRAM: 884,894,720 bytes
- Peak reserved VRAM: 960,495,616 bytes
- Checkpoint size: 208,968,659 bytes

Independent `nvidia-smi` samples during compute showed 92–94% GPU utilization,
about 1,878 MiB process memory, a maximum observed temperature of 45 C, and a
maximum observed power draw of about 187 W. Utilization drops corresponded to
startup, validation/checkpoint work, and process teardown.

## Checkpoints and generation

The run produced checksum-protected `best.pt`, `previous.pt`, and `latest.pt`.
Both the step-100 best checkpoint and final checkpoint were loaded by the
generation CLI. Output was recognizable as the synthetic smoke format but
remained repetitive and malformed in places. That limitation is expected after
only 1.024M repeated synthetic tokens and is not evidence of general language
ability.

The ignored runtime evidence is stored under:

- `logs/smoke-phase11/`
- `checkpoints/smoke-phase11/`
- `data/processed/smoke-million/`
- `data/tokenized/smoke-million/`
