---
model_name: Codexa v1
architecture: decoder-only Transformer
parameters: 921773568
context_length: 2048
vocabulary_size: 8192
language: en
license: other
---

# Codexa v1 Model Card

Codexa v1 is a 921,773,568-parameter decoder-only Transformer being trained
from random initialization with PyTorch. It uses 24 blocks, hidden size 1,536,
24 attention heads, SwiGLU feed-forward layers, RMSNorm, learned position
embeddings, and tied token/output embeddings.

## Training status

The rebuild has not produced an accepted checkpoint. The retained local
FineWeb-Edu artifact contains 883,814,184 training tokens from one pinned
shard. It is pipeline evidence and initial smoke data, not an adequate final
training budget for this model.

## Intended use

The first target is English text completion for educational model-development
research. It is not currently a chat assistant and must not be represented as
one. A future conversational stage is blocked until the base model passes
fixed coherence, repetition, termination, and held-out-loss gates.

## Limitations and risks

- No accepted model weights currently exist.
- FineWeb-Edu can contain errors, bias, private information, unsafe text,
  copyrighted material, and duplicates missed by exact matching.
- The tokenizer and corpus are English-focused and not specialized for code.
- Future outputs may be false, inconsistent, biased, repetitive, or malformed.
- No safety alignment or instruction-following ability is assumed.

## License

Repository source code is MIT licensed. Model-weight licensing remains `other`
until a release review accounts for all upstream dataset terms. FineWeb-Edu
requires ODC-By attribution.
