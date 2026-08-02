# Codexa v1

Codexa v1 is a from-scratch decoder-only Transformer base-model project built
with typed Python and PyTorch. The active target is the 921,773,568-parameter
configuration in `configs/1b.yaml`, with an 8,192-token vocabulary and a
2,048-token context window on one RTX 4080.

This repository currently covers base pretraining only: FineWeb-Edu download
and preparation, BPE tokenization, memory-mapped causal-LM data, mixed-precision
training, atomic checkpoints, text completion, and fixed-prompt evaluation.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

Use every script's `--help` before running it. Generated datasets, token data,
logs, and checkpoints are intentionally ignored by Git.

## Base data

The production corpus is the pinned `HuggingFaceFW/fineweb-edu` `sample-10BT`
configuration. The currently retained local artifact contains one prepared
shard and 883,814,184 training tokens. That shard is valid pipeline input, but
it is not the final training budget for a 1B-class model.

Download additional pinned shards:

```bash
.venv/bin/python -m scripts.download_fineweb_edu --help
```

Prepare them deterministically:

```bash
.venv/bin/python -m scripts.prepare_fineweb_edu --help
```

See `documentation/reference/DATASET.md` for the retained artifact's exact
revision, counts, and checksums.

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
`documentation/planning/PHASE_PLAN.md`.
