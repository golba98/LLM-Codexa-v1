# Phase 14 — 250M Prototype Run

## Result

The 248,565,504-parameter model completed a 50,003,968-token BF16 prototype
run on an NVIDIA GeForce RTX 4080. The run used `configs/250m_prototype.yaml`,
finished 763 optimizer steps and 12,208 micro-steps in 32 minutes 49 seconds,
and did not encounter an out-of-memory error, non-finite loss, or non-finite
gradient.

The first training loss was 9.1814, the final training loss was 2.4636, and
the minimum observed batch loss was 2.2769. Validation loss decreased at every
evaluation:

| Optimizer step | Tokens seen | Validation loss |
| ---: | ---: | ---: |
| 50 | 3,276,800 | 5.2435 |
| 100 | 6,553,600 | 3.9954 |
| 150 | 9,830,400 | 3.7037 |
| 200 | 13,107,200 | 3.5815 |
| 250 | 16,384,000 | 3.4796 |
| 300 | 19,660,800 | 3.3031 |
| 350 | 22,937,600 | 3.0724 |
| 400 | 26,214,400 | 2.8360 |
| 450 | 29,491,200 | 2.6395 |
| 500 | 32,768,000 | 2.4794 |
| 550 | 36,044,800 | 2.3779 |
| 600 | 39,321,600 | 2.2964 |
| 650 | 42,598,400 | 2.2382 |
| 700 | 45,875,200 | 2.1941 |
| 750 | 49,152,000 | 2.1651 |

## Runtime

- Mean logged throughput: 28,046 tokens/second
- Median logged throughput: 28,387 tokens/second
- End-to-end throughput including evaluation and checkpoints: about 25,396
  tokens/second
- Peak allocated VRAM: 10,569,384,960 bytes (9.84 GiB)
- Peak reserved VRAM: 11,406,409,728 bytes (10.62 GiB)
- Maximum reported pre-clipping gradient norm: 8.8901 at the first update
- Typical steady-state GPU utilization: 99%
- Observed temperature during training: approximately 60–62°C

The schedule warmed linearly from zero to `3e-4` over 50 optimizer steps, then
decayed smoothly to approximately `3e-5`. Loss remained stable at both the
warmup boundary and the low-learning-rate end of the run.

Checkpoints were saved every 100 optimizer steps, with atomic latest,
previous, best, and milestone files. Validation ran every 50 optimizer steps.

## Data and tokenizer checks

The pinned TinyStories preparation produced 1,799,248 training documents and
15,389 validation documents after cleaning and exact global deduplication.
The token dataset contains 384,882,826 training tokens and 3,075,250
validation tokens. Full SHA-256, binary-size, contiguous-index, and per-document
EOS checks passed after creation.

The byte-level BPE tokenizer has 8,192 entries, an unknown-token rate of zero
on the prepared corpus, and 4.1572 characters per token. This is efficient for
the selected English story corpus, but it is not evidence of good coverage for
code or broad multilingual text.

## Fixed-prompt samples

All samples below used the same prompt, seed 42, temperature 0.8, top-k 50,
top-p 0.95, and repetition penalty 1.1.

Prompt: `Once upon a time, there was a little girl named Lily`

- Step 100: fragmented output such as “The bird. The girl, the big...” with
  poor grammar.
- Step 300: recognizable story formatting and dialogue, but frequent broken
  clauses.
- Step 500: coherent story setup and character continuity, with semantic and
  grammatical mistakes.
- Step 700: multi-sentence story structure, dialogue, and a closing transition.
- Step 763: coherent local sentence flow and character continuity, though the
  story still contains implausible objects and unfinished reasoning.

Two additional final-checkpoint prompts produced complete UTF-8 text without
replacement characters or disallowed controls. Across the fixed samples,
repeated four-gram rate was zero. No pathological looping was observed.
Semantic consistency, pronoun use, and longer-range causality remain weak;
this is expected after only 50M tokens and should be measured again after the
intermediate run.

## Decision

No architecture change is required before Phase 15. The 250M configuration
fits safely, trains stably, keeps the GPU saturated, and shows a strictly
improving validation curve. The architecture in `configs/250m.yaml` is
approved for the intermediate run.

At the measured end-to-end rate, 500M tokens require approximately 5.5 hours,
1B tokens approximately 10.9 hours, 3B tokens approximately 32.8 hours, and
5B tokens approximately 54.7 hours. These estimates assume similar checkpoint
frequency, validation cost, hardware conditions, and sequence packing.
