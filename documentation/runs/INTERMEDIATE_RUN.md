# Phase 15 — Intermediate Training Run

## Result

The 248,565,504-parameter model completed a 500,037,632-token BF16
intermediate run on an NVIDIA GeForce RTX 4080. The run used
`configs/250m_intermediate.yaml`, finished 7,630 optimizer steps and 122,080
micro-steps, and did not encounter an out-of-memory error, non-finite loss,
non-finite gradient, CUDA failure, or thermal fault.

The nominal token count from fixed batch dimensions is 500,039,680. The
trainer recorded 500,037,632 actual tokens because one deterministic
DataLoader batch contained one example instead of two at an epoch boundary.
The trainer correctly counted the real batch rather than padding or
fabricating an example.

The first training loss was 9.1814 and the final training loss was 1.2015.
Scheduled validation loss decreased at every 250-step evaluation:

| Step | Tokens seen | Validation loss |
| ---: | ---: | ---: |
| 250 | 16,384,000 | 3.6861 |
| 500 | 32,768,000 | 2.9016 |
| 750 | 49,152,000 | 2.1070 |
| 1,000 | 65,536,000 | 1.7706 |
| 1,500 | 98,304,000 | 1.4959 |
| 2,000 | 131,072,000 | 1.3665 |
| 2,500 | 163,840,000 | 1.2827 |
| 3,000 | 196,608,000 | 1.2184 |
| 3,500 | 229,376,000 | 1.1687 |
| 4,000 | 262,144,000 | 1.1267 |
| 4,500 | 294,912,000 | 1.0888 |
| 5,000 | 327,680,000 | 1.0649 |
| 5,500 | 360,448,000 | 1.0369 |
| 6,000 | 393,213,952 | 1.0170 |
| 6,500 | 425,981,952 | 1.0036 |
| 7,000 | 458,749,952 | 0.9947 |
| 7,500 | 491,517,952 | 0.9862 |

The omitted 250-step rows follow the same strictly decreasing trend and remain
available in the ignored JSONL metrics log.

## Runtime and hardware

- Mean logged throughput: 28,823 tokens/second
- Median logged throughput: 28,947 tokens/second
- Peak allocated VRAM: 10,569,384,960 bytes (9.84 GiB)
- Peak reserved VRAM: 11,406,409,728 bytes (10.62 GiB)
- Maximum reported pre-clipping gradient norm: 8.8900 at the first update
- Long telemetry segment mean GPU utilization: 96.3%
- Maximum observed GPU temperature: 67°C
- Maximum observed GPU power draw: 265.15 W
- Maximum observed GPU memory use: 11,840 MiB

The run was deliberately interrupted at trusted milestone boundaries to test
resume behavior. Each resume verified the checkpoint checksum and restored
optimizer, scheduler, DataLoader, random-number-generator, and training state.
No completed optimizer step was lost.

## Checkpoints and evaluation

Milestone checkpoints were preserved every 1,000 optimizer steps. The final
checkpoint is:

```text
checkpoints/phase15-500m/latest.pt
SHA-256 9839bb307616b0d95bc0eadbc9cfadc76805baf17140e0c82d42e5a06051379e
```

The training-selected checkpoint is step 7,500:

```text
checkpoints/phase15-500m/best.pt
SHA-256 d2329c26ba2800a47168f93993cb2b178309c9452cbd56f7d3a89f0863a3f52d
```

Both files and their sidecars passed checksum and payload-state verification.
The fixed evaluation protocol compared the Phase 14 checkpoint and
intermediate milestones at steps 1,000 through 7,630. Under that protocol,
the final step-7,630 checkpoint was selected:

- Validation loss: 0.993063
- Validation perplexity: 2.69949
- Malformed generated characters: 0
- Mean repeated four-gram rate: reported in
  `logs/phase15-500m/evaluations/comparison_final.json`

Generated samples showed fluent local English story structure, dialogue,
character continuity, and no pathological loops. The narrow TinyStories
corpus did not produce reliable code completion, factual knowledge,
instruction following, or robust long-context recall. Those weaknesses are
data-domain limitations and are a primary reason for using the broader frozen
FineWeb-Edu corpus in Phase 16.

## Decision

The architecture and hyperparameters are stable enough for full pretraining.
The model learned meaningful short-story language behavior, validation
improved throughout the run, checkpoint resume was proven repeatedly, and
the measured hardware envelope remained safe.

Phase 16 remains gated on an independently mounted checkpoint-backup
destination and explicit confirmation of stable mains or UPS power. The full
run must not begin until `scripts/preflight_full_run.py` passes every check in
one final invocation.
