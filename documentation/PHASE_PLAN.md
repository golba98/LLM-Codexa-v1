# Codexa v1 — 250M LLM Phase Plan

Project path:

```text
/home/k9-vortex/Development/2-Python/31-LLM (Codexa v1)
```

Use this checklist to track progress from environment setup through full training and evaluation.

---

## Phase 0 — Project Setup

- [x] Confirm project folder is correct
- [x] Create Python virtual environment
- [x] Activate `.venv`
- [x] Upgrade `pip`
- [x] Install PyTorch
- [x] Install project dependencies
- [x] Confirm PyTorch imports successfully
- [x] Confirm CUDA is available
- [x] Confirm RTX 4080 is detected
- [x] Confirm approximately 16 GB VRAM is visible
- [x] Create initial project folders
- [x] Create `src/__init__.py`
- [x] Create `tests/__init__.py`
- [x] Create `.gitignore`
- [x] Save dependencies to `requirements.txt`
- [x] Initialize Git repository
- [x] Create first Git commit

### Phase 0 completion criteria

- [x] `python` runs from `.venv`
- [x] `torch.cuda.is_available()` returns `True`
- [x] RTX 4080 appears as the active GPU
- [x] Project structure exists and imports work

---

## Phase 1 — Small Transformer Model

- [x] Create `src/model.py`
- [x] Add `ModelConfig`
- [x] Add RMSNorm
- [x] Add causal self-attention
- [x] Add feed-forward network
- [x] Add Transformer block
- [x] Add token embeddings
- [x] Add position embeddings
- [x] Add final normalization
- [x] Add output projection
- [x] Add optional weight tying
- [x] Add parameter-counting function
- [x] Add input validation
- [x] Add sequence-length validation
- [x] Add model initialization
- [x] Add causal language-model loss

### Phase 1 completion criteria

- [x] Model imports without errors
- [x] Forward pass returns logits
- [x] Loss is finite
- [x] Output dimensions are correct
- [x] Parameter count is printed correctly

---

## Phase 2 — Model Tests

- [x] Create `tests/test_model.py`
- [x] Test model construction
- [x] Test valid input shape
- [x] Test logits shape
- [x] Test finite loss
- [x] Test context-length error
- [x] Test invalid head configuration
- [x] Test CPU forward pass
- [x] Test GPU forward pass
- [x] Test BF16 forward pass
- [x] Test parameter count
- [x] Run all tests successfully

### Phase 2 completion criteria

- [x] `python -m tests.test_model` passes
- [x] Model runs on the RTX 4080
- [x] No NaN or Inf values appear
- [x] Initial loss is close to `ln(vocab_size)`

---

## Phase 3 — Configuration System

- [x] Create `configs/smoke.yaml`
- [x] Create `configs/prototype.yaml`
- [x] Create `configs/250m.yaml`
- [x] Add YAML config loader
- [x] Validate required config fields
- [x] Validate model dimensions
- [x] Validate training settings
- [x] Print resolved configuration before training
- [x] Save a copy of the config with each checkpoint

### Phase 3 completion criteria

- [x] Model can be created entirely from YAML
- [x] Invalid configurations fail with clear errors
- [x] Every run records its exact configuration

---

## Phase 4 — Dataset Preparation

- [x] Decide the first small text dataset
- [x] Create `data/raw/`
- [x] Add sample training text
- [x] Add sample validation text
- [x] Create data-cleaning script
- [x] Normalize Unicode
- [x] Remove broken or empty documents
- [x] Remove duplicate documents
- [x] Record dataset source and license
- [x] Split data into train and validation sets
- [x] Add dataset statistics
- [x] Count characters
- [x] Count documents
- [x] Estimate token count
- [x] Save cleaned data to `data/processed/`

### Phase 4 completion criteria

- [x] Training and validation files exist
- [x] Data can be reproduced from scripts
- [x] Sources and licenses are documented
- [x] Dataset statistics are saved

---

## Phase 5 — Tokenizer

- [x] Choose BPE or Unigram tokenizer
- [x] Create tokenizer training script
- [x] Train tokenizer on representative data
- [x] Add `<pad>`
- [x] Add `<bos>`
- [x] Add `<eos>`
- [x] Add `<unk>`
- [x] Set tokenizer vocabulary size
- [x] Save tokenizer files
- [x] Test encode
- [x] Test decode
- [x] Test Unicode text
- [x] Test punctuation
- [x] Test numbers
- [x] Test code snippets
- [x] Test special token IDs
- [x] Measure average characters per token
- [x] Measure unknown-token rate

### Phase 5 completion criteria

- [x] Encoding and decoding work correctly
- [x] Special token IDs are stable
- [x] Tokenizer handles normal text and code
- [x] Tokenizer files can be reloaded

---

## Phase 6 — Tokenized Dataset Pipeline

- [x] Create tokenization script
- [x] Convert cleaned text to token IDs
- [x] Append EOS tokens between documents
- [x] Pack tokens into fixed-length sequences
- [x] Create training split
- [x] Create validation split
- [x] Save token arrays efficiently
- [x] Add memory-mapped loading
- [x] Create PyTorch dataset class
- [x] Create DataLoader
- [x] Add deterministic shuffling
- [x] Test batch shapes
- [x] Test labels are shifted correctly
- [x] Confirm no data leakage between splits
- [x] Measure data-loading throughput

### Phase 6 completion criteria

- [x] DataLoader returns valid input and label tensors
- [x] Batches load faster than the GPU consumes them
- [x] Training and validation splits remain separate
- [x] Dataset can resume deterministically

---

## Phase 7 — Training Loop

- [x] Create `src/train.py`
- [x] Move model to CUDA
- [x] Add AdamW optimizer
- [x] Add BF16 autocast
- [x] Add gradient accumulation
- [x] Add gradient clipping
- [x] Add learning-rate warmup
- [x] Add cosine learning-rate decay
- [x] Add training-loss logging
- [x] Add validation-loss logging
- [x] Add tokens-per-second logging
- [x] Add GPU-memory logging
- [x] Add step-time logging
- [x] Add NaN and Inf detection
- [x] Add random seed control
- [x] Add progress bar
- [x] Add graceful keyboard interruption
- [x] Add automatic cleanup after errors

### Phase 7 completion criteria

- [x] Loss decreases during training
- [x] GPU utilization is consistently high
- [x] Memory usage remains stable
- [x] Training can run for several hundred steps

---

## Phase 8 — Checkpointing and Resume

- [x] Create checkpoint save function
- [x] Save model state
- [x] Save optimizer state
- [x] Save scheduler state
- [x] Save training step
- [x] Save token count
- [x] Save random-number states
- [x] Save configuration
- [x] Save tokenizer reference
- [x] Create checkpoint load function
- [x] Add `--resume` support
- [x] Keep latest checkpoint
- [x] Keep best validation checkpoint
- [x] Keep previous known-good checkpoint
- [x] Add periodic milestone checkpoints
- [x] Test interrupted training
- [x] Test resumed training
- [x] Verify resumed loss is consistent

### Phase 8 completion criteria

- [x] Training resumes from the correct step
- [x] Optimizer and scheduler resume correctly
- [x] No checkpoint corruption occurs
- [x] At least two backup checkpoints exist

---

## Phase 9 — Text Generation

- [x] Create `src/generate.py`
- [x] Load model checkpoint
- [x] Load tokenizer
- [x] Add greedy decoding
- [x] Add temperature sampling
- [x] Add top-k sampling
- [x] Add top-p sampling
- [x] Add repetition penalty
- [x] Add maximum generation length
- [x] Add EOS stopping
- [x] Add prompt input
- [x] Add random seed option
- [x] Save generated samples
- [x] Test CPU generation
- [x] Test GPU generation

### Phase 9 completion criteria

- [x] Checkpoint can generate text
- [x] Generation stops correctly
- [x] Sampling options work
- [x] Outputs are saved for comparison

---

## Phase 10 — Tiny Overfit Test

- [x] Create a very small dataset
- [x] Train a 10M–30M model
- [x] Use short context length
- [x] Attempt to overfit a tiny batch
- [x] Confirm loss approaches a very low value
- [x] Generate memorized text
- [x] Save checkpoint
- [x] Reload checkpoint
- [x] Resume training
- [x] Verify results are reproducible

### Phase 10 completion criteria

- [x] Tiny model deliberately overfits
- [x] Loss and generation prove the pipeline works
- [x] Save, load, and resume all function correctly

---

## Phase 11 — Smoke Training Run

- [x] Train on 1M–5M tokens
- [x] Use the smoke configuration
- [x] Monitor training loss
- [x] Monitor validation loss
- [x] Monitor VRAM usage
- [x] Monitor GPU utilization
- [x] Record tokens per second
- [x] Record checkpoint size
- [x] Generate samples during training
- [x] Inspect samples for corruption
- [x] Fix all pipeline bugs

### Phase 11 completion criteria

- [x] Full pipeline completes without failure
- [x] Validation runs correctly
- [x] Checkpoints are usable
- [x] Generation is recognizable
- [x] No unresolved critical bugs remain

---

## Phase 12 — Design the 250M Architecture

- [x] Choose final vocabulary size
- [x] Choose hidden size
- [x] Choose number of layers
- [x] Choose number of attention heads
- [x] Choose intermediate size
- [x] Choose context length
- [x] Decide position encoding
- [x] Decide whether to tie embeddings
- [x] Calculate exact parameter count
- [x] Adjust architecture toward 250M
- [x] Estimate optimizer memory
- [x] Estimate activation memory
- [x] Confirm model fits in 16 GB VRAM
- [x] Document final architecture

### Phase 12 completion criteria

- [x] Exact parameter count is near 250M
- [x] Architecture fits on the RTX 4080
- [x] Configuration is saved in `configs/250m.yaml`

---

## Phase 13 — 250M VRAM and Speed Benchmark

- [x] Instantiate the 250M model
- [x] Run one forward pass
- [x] Run one backward pass
- [x] Test BF16
- [x] Test sequence length 512
- [x] Test sequence length 1024
- [x] Test sequence length 2048
- [x] Test micro-batch size 1
- [x] Test larger micro-batches
- [x] Test gradient accumulation
- [x] Test gradient checkpointing
- [x] Test `torch.compile`
- [x] Measure peak VRAM
- [x] Measure tokens per second
- [x] Measure step time
- [x] Choose stable training settings

### Phase 13 completion criteria

- [x] Stable batch and sequence settings are known
- [x] Realistic full-run duration is estimated
- [x] No out-of-memory errors at chosen settings

---

## Phase 14 — 250M Prototype Run

- [x] Train on 50M–100M tokens
- [x] Save regular checkpoints
- [x] Run validation regularly
- [x] Record training curves
- [x] Generate fixed prompt samples
- [x] Check for repetition
- [x] Check for malformed output
- [x] Check for data corruption
- [x] Review tokenizer quality
- [x] Review learning-rate behavior
- [x] Review GPU throughput
- [x] Estimate final training duration
- [x] Decide whether architecture changes are needed

### Phase 14 completion criteria

- [x] 250M model trains stably
- [x] Validation loss trends downward
- [x] Samples improve over time
- [x] Full-run configuration is approved

---

## Phase 15 — Intermediate Training Run

- [ ] Prepare 500M–1B tokens
- [x] Validate dataset quality
- [x] Confirm storage requirements
- [ ] Confirm checkpoint backup plan
- [x] Begin intermediate run
- [ ] Monitor thermals
- [ ] Monitor GPU stability
- [ ] Monitor loss curves
- [x] Generate milestone samples
- [x] Evaluate repetition
- [x] Evaluate coherence
- [x] Evaluate code completion
- [x] Evaluate long-context behavior
- [x] Compare checkpoints
- [ ] Select best checkpoint

### Phase 15 completion criteria

- [ ] Model shows meaningful language ability
- [ ] Dataset and hyperparameters are validated
- [ ] Full training run is justified

---

## Phase 16 — Full Pretraining Run

- [x] Prepare 3B–5B training tokens
- [x] Freeze final dataset version
- [x] Freeze tokenizer version
- [x] Freeze model configuration
- [x] Freeze training configuration
- [x] Verify disk space
- [ ] Verify backup storage
- [x] Verify system cooling
- [ ] Verify power stability
- [ ] Run final preflight test
- [ ] Start full training
- [ ] Save frequent checkpoints
- [ ] Save milestone checkpoints
- [ ] Run validation periodically
- [ ] Generate fixed prompt samples
- [ ] Record throughput
- [ ] Record downtime
- [ ] Resume safely after interruptions
- [ ] Complete target token count
- [ ] Save final checkpoint
- [ ] Save best validation checkpoint

### Phase 16 completion criteria

- [ ] Target token count is reached
- [ ] Final model and tokenizer are preserved
- [ ] Training logs and configs are complete
- [ ] Multiple valid checkpoints exist

---

## Phase 17 — Evaluation

- [ ] Create fixed evaluation prompt suite
- [ ] Measure validation perplexity
- [ ] Evaluate text coherence
- [ ] Evaluate repetition
- [ ] Evaluate factual consistency
- [ ] Evaluate code completion
- [ ] Evaluate instruction sensitivity
- [ ] Evaluate context retention
- [ ] Evaluate malformed output
- [ ] Evaluate memorization risk
- [ ] Evaluate dataset-category performance
- [ ] Compare prototype, intermediate, and final checkpoints
- [ ] Document strengths
- [ ] Document weaknesses
- [ ] Select release checkpoint

### Phase 17 completion criteria

- [ ] Evaluation report is complete
- [ ] Best checkpoint is selected
- [ ] Known limitations are documented

---

## Phase 18 — Optional Instruction Tuning

- [ ] Decide whether instruction tuning is needed
- [x] Prepare instruction dataset
- [x] Validate dataset licenses
- [x] Format chat templates
- [x] Add supervised fine-tuning script
- [x] Run small SFT test
- [ ] Evaluate instruction following
- [ ] Tune learning rate
- [ ] Train full SFT run
- [ ] Save base and instruction checkpoints separately
- [ ] Compare base and instruction models
- [x] Document chat format

### Phase 18 completion criteria

- [ ] Instruction model follows prompts reliably
- [ ] Base model remains preserved
- [ ] Chat template and usage are documented

---

## Phase 19 — Packaging and Release

- [ ] Clean repository
- [x] Update README
- [x] Add installation instructions
- [x] Add training instructions
- [x] Add generation instructions
- [x] Add architecture documentation
- [x] Add dataset documentation
- [x] Add tokenizer documentation
- [x] Add model limitations
- [x] Add license
- [ ] Add model card
- [ ] Export model weights
- [ ] Export tokenizer
- [ ] Add checksum files
- [ ] Tag release in Git
- [ ] Back up final artifacts

### Phase 19 completion criteria

- [ ] Another person can install and run the model
- [ ] Release files are complete
- [ ] Training process is reproducible
- [ ] Final model is safely backed up

---

# Current Progress

Update this section as work continues.

- [x] Phase 0 — Project Setup
- [x] Phase 1 — Small Transformer Model
- [x] Phase 2 — Model Tests
- [x] Phase 3 — Configuration System
- [x] Phase 4 — Dataset Preparation
- [x] Phase 5 — Tokenizer
- [x] Phase 6 — Tokenized Dataset Pipeline
- [x] Phase 7 — Training Loop
- [x] Phase 8 — Checkpointing and Resume
- [x] Phase 9 — Text Generation
- [x] Phase 10 — Tiny Overfit Test
- [x] Phase 11 — Smoke Training Run
- [x] Phase 12 — Design the 250M Architecture
- [x] Phase 13 — 250M VRAM and Speed Benchmark
- [x] Phase 14 — 250M Prototype Run
- [ ] Phase 15 — Intermediate Training Run
- [ ] Phase 16 — Full Pretraining Run
- [ ] Phase 17 — Evaluation
- [ ] Phase 18 — Optional Instruction Tuning
- [ ] Phase 19 — Packaging and Release

---

# Notes

Use this area for decisions, problems, benchmark results, and changes.

```text
Date:
Phase:
Decision or issue:
Result:
Next action:
```
