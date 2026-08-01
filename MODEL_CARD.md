---
model_name: Codexa v1
architecture: decoder-only Transformer
parameters: 920200704
context_length: 1024
vocabulary_size: 8192
language: en
license: other
---

# Codexa v1 Model Card

Codexa v1 is an educational decoder-only Transformer implemented and trained
from scratch with PyTorch. The current 1B-class base uses 24 Transformer blocks,
hidden size 1,536, 24 attention heads, SwiGLU feed-forward layers, RMSNorm,
learned position embeddings, and tied token/output embeddings.

## Training data

The current experiments use the pinned `roneneldan/TinyStories` dataset,
cleaned with Unicode NFKC normalization, control-character removal,
whitespace normalization, and exact SHA-256 deduplication. TinyStories declares
the CDLA-Sharing-1.0 license. See `documentation/reference/DATASET.md`.

## Intended use

The model is intended for educational research, pipeline validation, and
experimentation with small English story generation. It is not intended as a
production assistant or a source of reliable factual, legal, medical,
financial, safety, or security advice.

## Evaluation

The 50M-token prototype reduced validation loss from 5.2435 at 3.3M tokens to
2.1651 at 49.2M tokens. It produced increasingly coherent short-story text but
retained semantic, grammatical, causal, and long-range consistency errors.

The 500M-token intermediate run completed 7,630 optimizer updates without a
non-finite loss, CUDA failure, or thermal fault. Scheduled validation loss
decreased from 3.6861 at 16.4M tokens to 0.9862 at 491.5M tokens. A separate
fixed evaluation selected the final step-7,630 checkpoint with validation loss
0.9931 and no malformed generated characters. Short-story coherence and local
sentence flow improved substantially; code completion, factual reliability,
instruction following, and robust long-context recall remain weak. See
`documentation/runs/INTERMEDIATE_RUN.md`.

The inspected 1B-class base completed 15,000 pretraining steps and saw
491,520,000 tokens. Its best validation loss was 2.856837. These base-model
metrics do not establish chat quality; the checkpoint must complete canonical
template-3.0 SFT before it can be served as an assistant.

## Limitations and risks

- The training corpus is narrow, synthetic, English, and child-story focused.
- The model is not expected to complete code reliably or follow instructions.
- Outputs may be false, inconsistent, biased, repetitive, or malformed.
- Memorization testing is limited and cannot prove that training text will
  never be reproduced.
- No safety alignment or instruction tuning is assumed for the base model.
- The tokenizer has not been validated for broad multilingual or code use.

## License

Repository source code is MIT licensed. The model-weight license is intentionally
marked `other` until the final release review determines terms compatible with
the upstream dataset license. Do not infer MIT coverage for dataset files or
model weights.
