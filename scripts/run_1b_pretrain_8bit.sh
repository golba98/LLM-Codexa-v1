#!/usr/bin/env bash
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec ./.venv/bin/python scripts/train.py \
  --config configs/1b_1024.yaml \
  --train-token-file data/tokenized/fineweb-edu-1b-v1-context1024/train.bin \
  --validation-token-file data/tokenized/fineweb-edu-1b-v1-context1024/validation.bin \
  --token-manifest data/tokenized/fineweb-edu-1b-v1-context1024/token_data_manifest.json \
  --device cuda \
  --precision bf16 \
  --gradient-checkpointing \
  --optimizer adamw8bit \
  --run-name codexa-1b-pretrain-1024-8bit-v2 \
  --log-dir logs \
  --checkpoint-dir checkpoints \
  --max-validation-batches 4
