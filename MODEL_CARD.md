---
model_name: Codexa v1
architecture: decoder-only Transformer
parameters: 248565504
context_length: 2048
vocabulary_size: 8192
language: en
license: other
---

# Codexa v1 Model Card

Codexa v1 is an educational decoder-only Transformer implemented and trained
from scratch with PyTorch. The release candidate uses 34 Transformer blocks,
hidden size 768, 12 attention heads, SwiGLU feed-forward layers, RMSNorm,
learned position embeddings, and tied token/output embeddings.

## Training data

The current experiments use the pinned `roneneldan/TinyStories` dataset,
cleaned with Unicode NFKC normalization, control-character removal,
whitespace normalization, and exact SHA-256 deduplication. TinyStories declares
the CDLA-Sharing-1.0 license. See `documentation/DATASET.md`.

## Intended use

The model is intended for educational research, pipeline validation, and
experimentation with small English story generation. It is not intended as a
production assistant or a source of reliable factual, legal, medical,
financial, safety, or security advice.

## Evaluation

The 50M-token prototype reduced validation loss from 5.2435 at 3.3M tokens to
2.1651 at 49.2M tokens. It produced increasingly coherent short-story text but
retained semantic, grammatical, causal, and long-range consistency errors.
Intermediate and final evaluation results must be added before a weight
release is labeled final.

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
