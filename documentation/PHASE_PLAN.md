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

- [ ] Create tokenization script
- [ ] Convert cleaned text to token IDs
- [ ] Append EOS tokens between documents
- [ ] Pack tokens into fixed-length sequences
- [ ] Create training split
- [ ] Create validation split
- [ ] Save token arrays efficiently
- [ ] Add memory-mapped loading
- [ ] Create PyTorch dataset class
- [ ] Create DataLoader
- [ ] Add deterministic shuffling
- [ ] Test batch shapes
- [ ] Test labels are shifted correctly
- [ ] Confirm no data leakage between splits
- [ ] Measure data-loading throughput

### Phase 6 completion criteria

- [ ] DataLoader returns valid input and label tensors
- [ ] Batches load faster than the GPU consumes them
- [ ] Training and validation splits remain separate
- [ ] Dataset can resume deterministically

---

## Phase 7 — Training Loop

- [ ] Create `src/train.py`
- [ ] Move model to CUDA
- [ ] Add AdamW optimizer
- [ ] Add BF16 autocast
- [ ] Add gradient accumulation
- [ ] Add gradient clipping
- [ ] Add learning-rate warmup
- [ ] Add cosine learning-rate decay
- [ ] Add training-loss logging
- [ ] Add validation-loss logging
- [ ] Add tokens-per-second logging
- [ ] Add GPU-memory logging
- [ ] Add step-time logging
- [ ] Add NaN and Inf detection
- [ ] Add random seed control
- [ ] Add progress bar
- [ ] Add graceful keyboard interruption
- [ ] Add automatic cleanup after errors

### Phase 7 completion criteria

- [ ] Loss decreases during training
- [ ] GPU utilization is consistently high
- [ ] Memory usage remains stable
- [ ] Training can run for several hundred steps

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

- [ ] Create `src/generate.py`
- [ ] Load model checkpoint
- [ ] Load tokenizer
- [ ] Add greedy decoding
- [ ] Add temperature sampling
- [ ] Add top-k sampling
- [ ] Add top-p sampling
- [ ] Add repetition penalty
- [ ] Add maximum generation length
- [ ] Add EOS stopping
- [ ] Add prompt input
- [ ] Add random seed option
- [ ] Save generated samples
- [ ] Test CPU generation
- [ ] Test GPU generation

### Phase 9 completion criteria

- [ ] Checkpoint can generate text
- [ ] Generation stops correctly
- [ ] Sampling options work
- [ ] Outputs are saved for comparison

---

## Phase 10 — Tiny Overfit Test

- [ ] Create a very small dataset
- [ ] Train a 10M–30M model
- [ ] Use short context length
- [ ] Attempt to overfit a tiny batch
- [ ] Confirm loss approaches a very low value
- [ ] Generate memorized text
- [ ] Save checkpoint
- [ ] Reload checkpoint
- [ ] Resume training
- [ ] Verify results are reproducible

### Phase 10 completion criteria

- [ ] Tiny model deliberately overfits
- [ ] Loss and generation prove the pipeline works
- [ ] Save, load, and resume all function correctly

---

## Phase 11 — Smoke Training Run

- [ ] Train on 1M–5M tokens
- [ ] Use the smoke configuration
- [ ] Monitor training loss
- [ ] Monitor validation loss
- [ ] Monitor VRAM usage
- [ ] Monitor GPU utilization
- [ ] Record tokens per second
- [ ] Record checkpoint size
- [ ] Generate samples during training
- [ ] Inspect samples for corruption
- [ ] Fix all pipeline bugs

### Phase 11 completion criteria

- [ ] Full pipeline completes without failure
- [ ] Validation runs correctly
- [ ] Checkpoints are usable
- [ ] Generation is recognizable
- [ ] No unresolved critical bugs remain

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
- [ ] Confirm model fits in 16 GB VRAM
- [x] Document final architecture

### Phase 12 completion criteria

- [x] Exact parameter count is near 250M
- [ ] Architecture fits on the RTX 4080
- [x] Configuration is saved in `configs/250m.yaml`

---

## Phase 13 — 250M VRAM and Speed Benchmark

- [ ] Instantiate the 250M model
- [ ] Run one forward pass
- [ ] Run one backward pass
- [ ] Test BF16
- [ ] Test sequence length 512
- [ ] Test sequence length 1024
- [ ] Test sequence length 2048
- [ ] Test micro-batch size 1
- [ ] Test larger micro-batches
- [ ] Test gradient accumulation
- [ ] Test gradient checkpointing
- [ ] Test `torch.compile`
- [ ] Measure peak VRAM
- [ ] Measure tokens per second
- [ ] Measure step time
- [ ] Choose stable training settings

### Phase 13 completion criteria

- [ ] Stable batch and sequence settings are known
- [ ] Realistic full-run duration is estimated
- [ ] No out-of-memory errors at chosen settings

---

## Phase 14 — 250M Prototype Run

- [ ] Train on 50M–100M tokens
- [ ] Save regular checkpoints
- [ ] Run validation regularly
- [ ] Record training curves
- [ ] Generate fixed prompt samples
- [ ] Check for repetition
- [ ] Check for malformed output
- [ ] Check for data corruption
- [ ] Review tokenizer quality
- [ ] Review learning-rate behavior
- [ ] Review GPU throughput
- [ ] Estimate final training duration
- [ ] Decide whether architecture changes are needed

### Phase 14 completion criteria

- [ ] 250M model trains stably
- [ ] Validation loss trends downward
- [ ] Samples improve over time
- [ ] Full-run configuration is approved

---

## Phase 15 — Intermediate Training Run

- [ ] Prepare 500M–1B tokens
- [ ] Validate dataset quality
- [ ] Confirm storage requirements
- [ ] Confirm checkpoint backup plan
- [ ] Begin intermediate run
- [ ] Monitor thermals
- [ ] Monitor GPU stability
- [ ] Monitor loss curves
- [ ] Generate milestone samples
- [ ] Evaluate repetition
- [ ] Evaluate coherence
- [ ] Evaluate code completion
- [ ] Evaluate long-context behavior
- [ ] Compare checkpoints
- [ ] Select best checkpoint

### Phase 15 completion criteria

- [ ] Model shows meaningful language ability
- [ ] Dataset and hyperparameters are validated
- [ ] Full training run is justified

---

## Phase 16 — Full Pretraining Run

- [ ] Prepare 3B–5B training tokens
- [ ] Freeze final dataset version
- [ ] Freeze tokenizer version
- [ ] Freeze model configuration
- [ ] Freeze training configuration
- [ ] Verify disk space
- [ ] Verify backup storage
- [ ] Verify system cooling
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
- [ ] Prepare instruction dataset
- [ ] Validate dataset licenses
- [ ] Format chat templates
- [ ] Add supervised fine-tuning script
- [ ] Run small SFT test
- [ ] Evaluate instruction following
- [ ] Tune learning rate
- [ ] Train full SFT run
- [ ] Save base and instruction checkpoints separately
- [ ] Compare base and instruction models
- [ ] Document chat format

### Phase 18 completion criteria

- [ ] Instruction model follows prompts reliably
- [ ] Base model remains preserved
- [ ] Chat template and usage are documented

---

## Phase 19 — Packaging and Release

- [ ] Clean repository
- [ ] Update README
- [ ] Add installation instructions
- [ ] Add training instructions
- [ ] Add generation instructions
- [ ] Add architecture documentation
- [ ] Add dataset documentation
- [ ] Add tokenizer documentation
- [ ] Add model limitations
- [ ] Add license
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
- [ ] Phase 6 — Tokenized Dataset Pipeline
- [ ] Phase 7 — Training Loop
- [x] Phase 8 — Checkpointing and Resume
- [ ] Phase 9 — Text Generation
- [ ] Phase 10 — Tiny Overfit Test
- [ ] Phase 11 — Smoke Training Run
- [ ] Phase 12 — Design the 250M Architecture
- [ ] Phase 13 — 250M VRAM and Speed Benchmark
- [ ] Phase 14 — 250M Prototype Run
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
