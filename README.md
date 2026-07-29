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

## Tokenizer and token data

Train the 8,192-entry byte-level BPE tokenizer without loading the whole corpus:

```bash
.venv/bin/python -m scripts.train_tokenizer \
  data/processed/tinystories/train.jsonl \
  data/processed/tinystories/validation.jsonl \
  --output-dir checkpoints/tokenizer-tinystories \
  --vocab-size 8192 \
  --streaming \
  --overwrite
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
- `configs/250m.yaml`: long-run architecture and baseline training settings.

Checkpoints contain model, optimizer, scheduler, RNG, configuration, and
training state. Resume only trusted local checkpoints:

```bash
.venv/bin/python -m scripts.train [same options] \
  --resume checkpoints/RUN_NAME/latest.pt
```

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

## Limitations

- TinyStories is a narrow synthetic English story corpus. A model trained only
  on it is not a general assistant, factual reference, or code model.
- Generated text can be incorrect, contradictory, repetitive, biased, or
  inappropriate. It must not be used for high-stakes decisions.
- The 8,192-token tokenizer was optimized for this corpus and has not been
  validated for broad multilingual or source-code coverage.
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
