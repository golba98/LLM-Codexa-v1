# Phase 17 — Evaluation Report

## Scope

The fixed prompt suite was run on the preserved 250M base, the earlier Phase
16 chat checkpoint, and the HF SFT checkpoint. Reports are stored under
`logs/phase17-evaluation/`.

The large tokenized validation binaries were intentionally removed during disk
cleanup. Validation loss and perplexity below therefore use the recorded
`best_validation_loss` saved inside each checkpoint; generation and quality
metrics were run live on CUDA.

## Validation metrics

| Checkpoint | Validation loss | Perplexity | Best step |
| --- | ---: | ---: | ---: |
| Phase 16 250M base | 2.5230 | 12.4664 | 45,500 |
| Phase 16 chat SFT | 1.5652 | 4.7835 | 900 |
| HF chat SFT | 1.9263 | 6.8641 | 8,000 |

## Fixed-prompt results

The seven simple chat prompts measured causal consistency, factual
consistency, code completion, exact-format instruction following, context
retention, story coherence, and a memorization probe.

| Checkpoint | Expected outcomes | Exact-format | Mean repeated n-gram rate | Malformed characters |
| --- | ---: | ---: | ---: | ---: |
| Phase 16 chat SFT | 3/5 | 1/1 | 0.0032 | 0 |
| HF chat SFT | 2/5 | 1/1 | 0.0000 | 0 |

The HF chat checkpoint also matched `sapphire compass` in the separate
1,536-token long-context probe. That is useful evidence of context recall, but
it does not outweigh the lower overall fixed-suite score.

## Decision

The strongest currently preserved chat checkpoint is:

```text
checkpoints/codexa-v1-chat-phase16/codexa-v1-chat-phase16/best.pt
```

This is not a release-quality general assistant. Both chat checkpoints remain
weak on code completion, factual consistency, and natural conversation. The
next 1B-from-scratch run is justified by this evidence; its Phase 21 SFT must
repeat this evaluation after using the cleaned conversation dataset.
