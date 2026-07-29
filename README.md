# Codexa v1

Codexa v1 is a from-scratch decoder-only Transformer project built with typed
Python and PyTorch. The repository covers reproducible text preparation,
byte-level BPE tokenization, memory-mapped causal-LM datasets, mixed-precision
training, atomic checkpoints, generation, evaluation, and release export.

The current full architecture has 248,565,504 parameters, an 8,192-token
vocabulary, and a 2,048-token context window. It is designed to fit and train
on a single 16 GB NVIDIA GPU.

## Requirements

- Python 3.14
- PyTorch 2.13 with CUDA 13 for GPU training
- An NVIDIA GPU with BF16 support for the documented training configuration
- About 12 GB of free VRAM for the stable micro-batch
- Substantial local storage for source data and checkpoints

Create and populate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The exact environment used for the verified runs is Fedora Linux, an NVIDIA
GeForce RTX 4080, PyTorch `2.13.0+cu130`, and CUDA BF16.

## Verify the project

Every test module uses ordinary assertions and runs without pytest:

```bash
for module in $(find tests -maxdepth 1 -name 'test_*.py' -printf '%f\n' \
  | sed 's/\.py$//' | sort); do
  .venv/bin/python -m "tests.${module}"
done
.venv/bin/python -m compileall src tests scripts
```

## Data preparation

The small tracked fixture can verify the local cleaning pipeline:

```bash
.venv/bin/python -m scripts.prepare_dataset \
  tests/fixtures/data/sample.txt \
  --output-dir data/processed/sample \
  --dataset-name codexa-sample \
  --license original-test-fixture \
  --validation-ratio 0.05 \
  --seed 42
```

The production experiment uses a pinned TinyStories revision:

```bash
.venv/bin/python -m scripts.download_tinystories
.venv/bin/python -m scripts.prepare_tinystories --overwrite
```

Raw and processed corpora are intentionally ignored by Git. See
`documentation/DATASET.md` for provenance, license, checksums, and limitations.

Prepare the pinned four-shard FineWeb-Edu candidate for the full run:

```bash
.venv/bin/python -m scripts.download_fineweb_edu --shard-count 4
.venv/bin/python -m scripts.prepare_fineweb_edu \
  data/raw/fineweb-edu-10bt/sample/10BT/000_00000.parquet \
  data/raw/fineweb-edu-10bt/sample/10BT/001_00000.parquet \
  data/raw/fineweb-edu-10bt/sample/10BT/002_00000.parquet \
  data/raw/fineweb-edu-10bt/sample/10BT/003_00000.parquet \
  --output-dir data/processed/fineweb-edu \
  --validation-ratio 0.005 \
  --seed 42
```

## Tokenizer and token data

Train the 8,192-entry byte-level BPE tokenizer without loading the whole corpus:

```bash
.venv/bin/python -m scripts.train_tokenizer \
  data/processed/tinystories/train.jsonl \
  data/processed/tinystories/validation.jsonl \
  --output-dir checkpoints/tokenizer-tinystories \
  --vocab-size 8192 \
  --streaming
```

Create memory-mapped token streams:

```bash
.venv/bin/python -m scripts.tokenize_dataset \
  --train-jsonl data/processed/tinystories/train.jsonl \
  --validation-jsonl data/processed/tinystories/validation.jsonl \
  --tokenizer checkpoints/tokenizer-tinystories/tokenizer.json \
  --output-dir data/tokenized/tinystories \
  --model-vocab-size 8192 \
  --context-length 2048 \
  --overwrite
```

The final broad-corpus tokenizer and token stream use separate paths so the
TinyStories experiment remains reproducible:

```bash
.venv/bin/python -m scripts.train_tokenizer \
  data/processed/fineweb-edu/train.jsonl \
  data/processed/fineweb-edu/validation.jsonl \
  --output-dir checkpoints/tokenizer-fineweb-edu \
  --vocab-size 8192 \
  --min-frequency 2 \
  --streaming

.venv/bin/python -m scripts.tokenize_dataset \
  --train-jsonl data/processed/fineweb-edu/train.jsonl \
  --validation-jsonl data/processed/fineweb-edu/validation.jsonl \
  --tokenizer checkpoints/tokenizer-fineweb-edu/tokenizer.json \
  --output-dir data/tokenized/fineweb-edu \
  --model-vocab-size 8192 \
  --context-length 2048 \
  --encoding-batch-size 128
```

## Training

Run the small smoke configuration:

```bash
.venv/bin/python -m scripts.train \
  --config configs/smoke.yaml \
  --train-token-file data/tokenized/train.bin \
  --validation-token-file data/tokenized/validation.bin \
  --token-manifest data/tokenized/token_data_manifest.json \
  --device cuda \
  --precision bf16 \
  --max-steps 2 \
  --no-validation \
  --run-name smoke
```

The 250M configurations are:

- `configs/250m_prototype.yaml`: verified 50M-token experiment.
- `configs/250m_intermediate.yaml`: 500M-token intermediate run.
- `configs/250m_full.yaml`: 3,000,041,472-token full run
  (45,777 optimizer steps).
- `configs/250m.yaml`: long-run architecture and baseline training settings.

The frozen production inputs, exact token counts, storage projection, and
strict preflight command are documented in `documentation/FULL_RUN.md`.

Checkpoints contain model, optimizer, scheduler, RNG, configuration, and
training state. Resume only trusted local checkpoints:

```bash
.venv/bin/python -m scripts.train [same options] \
  --resume checkpoints/RUN_NAME/latest.pt
```

Record independent GPU temperature, power, utilization, clocks, and memory
telemetry during long runs:

```bash
.venv/bin/python -m scripts.monitor_gpu \
  --output logs/RUN_NAME/gpu_metrics.jsonl \
  --interval-seconds 30
```

Each JSONL record is flushed immediately. Use `--append` only when continuing
the same run's existing telemetry file.

Summarize it with
`.venv/bin/python -m scripts.summarize_gpu logs/RUN_NAME/gpu_metrics.jsonl`.

## Release export and backup

Export only inference artifacts from the selected checkpoint:

```bash
.venv/bin/python -m scripts.export_release \
  --checkpoint checkpoints/RUN_NAME/best.pt \
  --tokenizer checkpoints/tokenizer-tinystories/tokenizer.json \
  --output-dir releases/codexa-v1
```

The release contains safetensors weights, tokenizer, model configuration,
training state, model card, license, manifest, and `SHA256SUMS`. Release
directories and model weights remain ignored by Git.

A real backup must use an independently mounted filesystem. The backup command
rejects a same-filesystem destination by default:

```bash
.venv/bin/python -m scripts.backup_artifacts \
  releases/codexa-v1 \
  --destination /path/on/independent/storage/codexa-v1
```

The backup is written atomically and every copied file is verified by SHA-256.

## Generation

```bash
.venv/bin/python -m scripts.generate \
  --checkpoint checkpoints/RUN_NAME/best.pt \
  --tokenizer checkpoints/tokenizer-tinystories/tokenizer.json \
  --prompt "Once upon a time" \
  --device cuda \
  --max-new-tokens 128 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95 \
  --repetition-penalty 1.1
```

No BOS or EOS token is silently added to prompts. Generation stops at EOS or
the context limit.

After exporting a release directory, generation no longer requires the trusted
PyTorch training checkpoint:

```bash
.venv/bin/python -m scripts.generate \
  --release-dir releases/codexa-v1 \
  --prompt "Once upon a time" \
  --device cuda \
  --max-new-tokens 128
```

The loader verifies every entry in `SHA256SUMS` before loading safetensors.

## Evaluation

`configs/evaluation_prompts.json` defines fixed prompts for coherence,
causality, factual consistency, code completion, instruction sensitivity,
context retention, and memorization probes.

```bash
.venv/bin/python -m scripts.evaluate_checkpoint \
  --checkpoint checkpoints/RUN_NAME/best.pt \
  --tokenizer checkpoints/tokenizer-tinystories/tokenizer.json \
  --validation-token-file data/tokenized/tinystories/validation.bin \
  --token-manifest data/tokenized/tinystories/token_data_manifest.json \
  --reference-jsonl data/processed/tinystories/train.jsonl \
  --device cuda \
  --output logs/RUN_NAME/evaluation.json \
  --overwrite
```

Compare two or more reports with the fixed selection rule:

```bash
.venv/bin/python -m scripts.compare_evaluations \
  logs/prototype/evaluation.json \
  logs/intermediate/step_2000_evaluation.json \
  logs/intermediate/best_evaluation.json \
  --output logs/tinystories_validation_comparison.json \
  --overwrite
```

Validation ranking requires identical tokenizer, prompt-suite, and validation
token checksums. To compare models trained with different final tokenizers,
rank the same fixed prompt suite by expected outcomes and output quality:

```bash
.venv/bin/python -m scripts.compare_evaluations \
  logs/prototype/evaluation.json \
  logs/intermediate/evaluation.json \
  logs/final/evaluation.json \
  --selection-metric quality \
  --output logs/cross_tokenizer_quality_comparison.json \
  --overwrite
```

The comparison preserves per-category metrics so the selected checkpoint can
still be reviewed for regressions that an aggregate rank would hide.

For a fair base-versus-SFT instruction comparison, evaluate both checkpoints
with the same documented chat template:

```bash
.venv/bin/python -m scripts.evaluate_checkpoint \
  --checkpoint checkpoints/RUN_NAME/best.pt \
  --tokenizer checkpoints/FINAL_TOKENIZER/tokenizer.json \
  --prompts configs/instruction_evaluation_prompts.json \
  --instruction-template \
  --device cuda \
  --output logs/RUN_NAME/instruction_evaluation.json \
  --overwrite
```

## Optional instruction tuning

The base model and instruction model use separate run names and checkpoint
directories. Download the pinned CC-BY-SA Dolly dataset:

```bash
.venv/bin/python -m scripts.download_dolly
```

After selecting the final base checkpoint, run response-masked supervised
fine-tuning:

```bash
.venv/bin/python -m scripts.train_sft \
  --config configs/250m_sft.yaml \
  --base-checkpoint checkpoints/BASE_RUN/best.pt \
  --tokenizer checkpoints/FINAL_TOKENIZER/tokenizer.json \
  --instruction-jsonl \
    data/raw/databricks-dolly-15k/databricks-dolly-15k.jsonl \
  --device cuda \
  --precision bf16 \
  --run-name codexa-v1-sft \
  --checkpoint-dir checkpoints \
  --overwrite-log
```

See `documentation/CHAT_TEMPLATE.md` for the exact template, target masking,
truncation behavior, and derivative-license warning.

Generate from an instruction-tuned release with automatic template formatting:

```bash
.venv/bin/python -m scripts.generate \
  --release-dir releases/codexa-v1-instruct \
  --instruction "Write exactly three colors, separated by commas." \
  --device cuda \
  --max-new-tokens 64
```

## Limitations

- The Phase 14–15 TinyStories checkpoints are narrow synthetic English story
  models, not general assistants, factual references, or code models.
- FineWeb-Edu broadens educational and general web coverage for Phase 16, but
  the base model is still not instruction-tuned, safety-aligned, or reliable
  for factual, code, professional, or high-stakes use.
- Generated text can be incorrect, contradictory, repetitive, biased, or
  inappropriate. It must not be used for high-stakes decisions.
- The 8,192-token FineWeb-Edu tokenizer has zero unknown tokens on the frozen
  corpus but is not optimized for broad multilingual or source-code coverage.
- Exact bitwise reproducibility is not guaranteed across different GPUs,
  drivers, CUDA versions, and PyTorch releases.
- Checkpoints are loaded through PyTorch serialization and must be treated as
  trusted local files.

See `documentation/ARCHITECTURE.md`, `documentation/PROTOTYPE_RUN.md`, and
`documentation/PHASE_PLAN.md` for design and measured evidence.

## License

Project source code is provided under the MIT License. Dataset and released
weight terms are documented separately because upstream dataset licenses still
apply.
