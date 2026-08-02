# Codexa v1 — Base Model Rebuild Plan

Codexa will restart as a decoder-only base language model trained from random
weights on existing, licensed text datasets. We are not writing or generating a
new pretraining dataset. We are also not doing chat fine-tuning until the base
checkpoint can produce coherent English continuations.

The target is the approximately 1-billion-parameter architecture in
`configs/1b.yaml`, trained locally on one RTX 4080. The cheap validation stages
use this same 1B architecture with very short runs; there is no smaller-model
training stage or prerequisite.

## Non-goals

- No Google Colab, Google Drive, hosted-notebook, or remote-session workflow.
- No hand-authored, templated, or synthetic corpus presented as base data.
- No chat template, assistant persona, SFT, preference training, tools, RAG, or
  OpenAI-compatible serving during base pretraining.
- No claim that a base model is a chat assistant. Its first job is coherent text
  completion; chat behavior comes later.
- No reuse of old model weights. The approved run starts from a new random seed.

## Existing data we will use

The downloaded sources have distinct jobs:

- General and educational language: ten pinned shards from
  `HuggingFaceFW/fineweb-edu`, `sample-10BT`.
- Encyclopedic and factual prose: the complete pinned English
  `wikimedia/wikipedia` `20231101.en` snapshot.
- Later conversational SFT: pinned `HuggingFaceH4/ultrachat_200k` `train_sft`
  and `test_sft`, plus pinned `OpenAssistant/oasst1` train and validation.

FineWeb-Edu and Wikipedia form the base-pretraining candidate. UltraChat and
OASST1 remain role-structured and separate until the base quality gate passes;
mixing flattened dialogue into base text would discard the supervision needed
to teach turn taking. Training and validation remain reproducible, and all
generated artifacts remain ignored by Git.

## Stage 1 — Freeze the base specification

- [ ] Confirm `configs/1b.yaml` is the rebuild architecture and record its exact
  parameter count.
- [ ] Confirm the run creates a newly initialized model and cannot silently
  resume an old checkpoint.
- [ ] Record the exact FineWeb-Edu and Wikipedia manifests, mixture weights,
  tokenizer checksum, token count, code commit, config, seed, and environment.
- [ ] Keep the vocabulary at 8,192 and context at 2,048 for the first rebuild;
  change neither while diagnosing language quality.
- [ ] Create a new run name and empty output directory dedicated to the rebuild.

### Exit gate

One written run manifest identifies every input. The command fails rather than
loading an old checkpoint or mismatched tokenizer.

## Stage 2 — Prove the training path cheaply

- [ ] Run the existing tests with `python -m pytest`.
- [ ] Run the tiny-overfit test and verify the model memorizes its small fixture.
- [ ] Run a short FineWeb-Edu smoke job from random weights.
- [ ] Verify causal labels, document boundaries, validation separation,
  checkpoint save/resume, and deterministic seed behavior.
- [ ] Generate fixed samples before training and after the smoke job.

### Exit gate

Loss is finite and falls, the tiny fixture can be overfit, checkpoints resume
correctly, and fixed-prompt output changes in the expected direction. Any
failure is fixed here before a long run starts.

## Stage 3 — Train the 1B base model

- [ ] Train only on the prepared FineWeb-Edu and Wikipedia base mixture.
- [ ] Start from random weights; do not initialize from any older checkpoint.
- [ ] Use BF16, gradient accumulation, clipping, warmup, and cosine decay from
  the existing local training loop.
- [ ] Save `latest` for recovery and retain milestone checkpoints for comparison.
- [ ] Log training loss, validation loss, learning rate, gradient norm, tokens
  processed, throughput, GPU memory, and wall time.
- [ ] Evaluate and generate the same fixed prompt suite at every milestone.
- [ ] Stop early for NaN/Inf, persistent validation regression, broken samples,
  tokenizer mismatch, or corrupted checkpoints.

This stage is measured in processed tokens, not merely optimizer steps. The
first complete attempt should consume the prepared corpus once. Additional
tokens or epochs require evidence from validation and sample quality.

## Stage 4 — Base language quality gate

The model must pass deterministic checks before any chat work begins:

- [ ] Complete ordinary English prose with readable grammar and topic continuity.
- [ ] Continue a short story without immediately collapsing into repetition.
- [ ] Continue factual/educational prose in the style of the prompt without
  pretending that factual accuracy has been established.
- [ ] Preserve basic formatting for lists, headings, quotations, and paragraphs.
- [ ] Avoid premature EOS, endless loops, copied prompt fragments, and token
  garbage across the fixed evaluation set.
- [ ] Beat the untrained checkpoint and earlier milestones on held-out loss.
- [ ] Pass repetition, distinct-token, and completion-length diagnostics.
- [ ] Record failures as failures; sampling changes cannot be used to hide a bad
  checkpoint.

Use greedy decoding as a reproducible health check plus one fixed sampling
configuration for readability. A checkpoint passes only when the behavior is
repeatable across a fixed, versioned prompt set.

### Exit gate

The selected checkpoint is a usable text-completion base model: it can produce
several coherent paragraphs on multiple unseen prompts without systemic
repetition or collapse. If it fails, investigate data, tokenization, labels,
optimization, and training duration before adding chat data.

## Stage 5 — Select and preserve the 1B base

- [ ] Compare 1B milestone checkpoints using held-out loss and the fixed prompt
  suite rather than automatically choosing the final step.
- [ ] Preserve the selected base checkpoint, tokenizer, manifest, metrics, and
  generated evaluation samples together.
- [ ] Document the training-token count, runtime, hardware, limitations, data
  provenance, and exact quality-gate results.
- [ ] Keep the base checkpoint immutable before beginning any chat training.

## Stage 6 — Chat comes after the base

This stage is deliberately blocked until Stage 4 passes. The source data is
downloaded now so its provenance and format can be audited before training.

- [ ] Preserve the passing base checkpoint unchanged.
- [ ] Audit UltraChat's synthetic-generation caveat and OASST1's human
  conversation-tree structure, quality, and licenses.
- [ ] Build a separate, versioned SFT mixture from accepted examples without
  losing message roles or assistant-only label boundaries.
- [ ] Train assistant-only labels with the repository chat protocol.
- [ ] Evaluate multi-turn retention, instruction following, factuality,
  repetition, stop behavior, and base-capability regression.
- [ ] Expose a checkpoint through the chat server only after those gates pass.

## Immediate next actions

1. Verify the four downloaded source manifests and inspect their schemas.
2. Implement deterministic FineWeb-Edu plus Wikipedia preparation and mixture
   accounting; keep dialogue data outside this base path.
3. Re-train/freeze the tokenizer on the approved base mixture and rebuild token
   artifacts.
4. Run tests, a tiny-step 1B validation job, and a short base-mixture smoke job.
5. Review the evidence, then launch the 1B base pretraining run locally.

The rebuild is successful when Codexa first works as a coherent base text model.
Chat behavior and serving are later projects, not shortcuts around that
requirement.
