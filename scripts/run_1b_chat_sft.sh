#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_name="${1:-codexa-1b-chat-v3}"
metrics="logs/${run_name}/train_metrics.jsonl"

kitty --detach \
  --title "Codexa v3 · 920M Training" \
  --directory "$PWD" \
  zsh -lc "PYTHONPATH=. .venv/bin/python scripts/watch_chat_training.py --metrics '$metrics' --max-steps 1000; exec zsh"

exec env PYTHONPATH=. .venv/bin/python -m scripts.train_sft \
  --config configs/1b_1024_chat_sft.yaml \
  --base-checkpoint checkpoints/codexa-1b-pretrain-1024-8bit-v3/best.pt \
  --tokenizer checkpoints/tokenizer-fineweb-edu-chat-v3/tokenizer.json \
  --instruction-jsonl data/processed/codexa-chat-v3/chat.jsonl \
  --validation-ratio 0.05 \
  --device cuda \
  --precision bf16 \
  --optimizer adamw8bit \
  --max-validation-batches 8 \
  --log-dir logs \
  --run-name "$run_name" \
  --checkpoint-dir checkpoints
