# Codexa v1

Codexa v1 is a from-scratch decoder-only Transformer base-model project built
with typed Python and PyTorch. The active target is the 921,773,568-parameter
configuration in `configs/1b.yaml`, with an 8,192-token vocabulary and a
2,048-token context window on one RTX 4080.

This repository currently covers base pretraining: licensed general and
encyclopedic corpus download, FineWeb-Edu preparation, BPE tokenization,
memory-mapped causal-LM data, mixed-precision training, atomic checkpoints,
text completion, and fixed-prompt evaluation. Existing conversational corpora
are downloaded separately and retain their role structure for later SFT.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

Use every script's `--help` before running it. Generated datasets, token data,
logs, and checkpoints are intentionally ignored by Git.

## Training data

The base corpus combines pinned FineWeb-Edu general/educational text with a
pinned English Wikipedia snapshot. Ten FineWeb-Edu `sample-10BT` shards and
the complete `20231101.en` Wikipedia snapshot are the first downloaded source
set. The currently prepared/tokenized artifact still contains only one
FineWeb-Edu shard and 883,814,184 training tokens; regenerate it after the
multi-source preparation path is complete.

Download additional pinned shards:

```bash
.venv/bin/python -m scripts.download_fineweb_edu --help
.venv/bin/python -m scripts.download_language_corpora --help
```

Prepare them deterministically:

```bash
.venv/bin/python -m scripts.prepare_fineweb_edu --help
```

UltraChat 200k and OASST1 are downloaded as later conversational training
sources. They must not be flattened into the base token stream or treated as
Wikipedia/general knowledge. See `documentation/reference/DATASET.md` for
revisions, roles, licenses, and the retained artifact's checksums.

## Tokenizer and token data

```bash
.venv/bin/python -m scripts.train_tokenizer --help
.venv/bin/python -m scripts.tokenize_dataset --help
.venv/bin/python -m scripts.inspect_tokenizer --help
.venv/bin/python -m scripts.inspect_token_data --help
```

The active tokenizer is a byte-level BPE tokenizer with 8,192 entries. Special
token IDs are fixed at `<pad>=0`, `<bos>=1`, `<eos>=2`, and `<unk>=3`.

## Validate before a long run

```bash
.venv/bin/python -m scripts.run_tiny_overfit --help
.venv/bin/python -m scripts.preflight_full_run --help
```

The long run must start from random weights in a new output directory. Resume
is allowed only after that run has produced its own trusted checkpoint.

## Train the 1B base

```bash
.venv/bin/python -m scripts.train \
  --config configs/1b.yaml \
  --train-token-file data/tokenized/fineweb-edu-1b-v1/train.bin \
  --validation-token-file data/tokenized/fineweb-edu-1b-v1/validation.bin \
  --token-manifest data/tokenized/fineweb-edu-1b-v1/token_data_manifest.json \
  --device cuda \
  --precision bf16 \
  --gradient-checkpointing \
  --optimizer adamw8bit \
  --run-name codexa-1b-base-rebuild
```

The retained one-shard token stream is suitable for smoke and throughput
validation. Do not treat a run over only that stream as a finished 1B base.

## Generate and evaluate

```bash
.venv/bin/python -m scripts.generate \
  --checkpoint checkpoints/codexa-1b-base-rebuild/best.pt \
  --tokenizer checkpoints/tokenizer-fineweb-edu/tokenizer.json \
  --prompt "The purpose of education is" \
  --device cuda \
  --greedy

.venv/bin/python -m scripts.evaluate_checkpoint \
  --checkpoint checkpoints/codexa-1b-base-rebuild/best.pt \
  --tokenizer checkpoints/tokenizer-fineweb-edu/tokenizer.json \
  --validation-token-file data/tokenized/fineweb-edu-1b-v1/validation.bin \
  --token-manifest data/tokenized/fineweb-edu-1b-v1/token_data_manifest.json \
  --output logs/codexa-1b-base-rebuild/evaluation.json \
  --device cuda
```

The base checkpoint is accepted only when fixed unseen prompts produce
coherent continuations without systemic repetition or collapse. Assistant and
multi-turn training are separate future work after that gate passes.

The active execution checklist is
`documentation/planning/PHASE_PLAN.md`. For a plain-language explanation of
what each dataset teaches and why base training and chat fine-tuning are
separate, read `documentation/planning/TRAINING_DATA_PLAN.md`.
