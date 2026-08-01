# 250M RTX 4080 Benchmark

## Environment

- Model parameters: 248,565,504
- GPU: NVIDIA GeForce RTX 4080
- Visible VRAM: 16,748,380,160 bytes
- PyTorch: 2.13.0+cu130
- CUDA runtime: 13.0
- Precision: BF16 autocast with FP32 parameters

## Results

| Case | Tokens/s | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: |
| Context 512, batch 1 | 2,314 | 2.25 GB | 2.38 GB |
| Context 1,024, batch 1 | 23,562 | 3.03 GB | 3.04 GB |
| Context 2,048, batch 1 | 26,462 | 4.53 GB | 4.73 GB |
| Context 512, batch 2 | 27,233 | 3.03 GB | 3.04 GB |
| Accumulation 2, context 512 | 13,637 | 3.32 GB | 3.44 GB |
| Checkpointed context 2,048 | 3,808 | 2.14 GB | 2.35 GB |
| AdamW step, context 2,048, batch 1 | 15,363 | 5.05 GB | 5.48 GB |
| AdamW step, context 2,048, batch 2 | 22,801 | 7.45 GB | 7.61 GB |
| `torch.compile`, AOT eager, context 512 | 80 | 2.09 GB | 2.46 GB |

The first 512-token eager case includes CUDA/kernel warm-up and is not a
steady-state comparison. Activation checkpointing substantially reduces
memory but is slower, so it is available as a fallback rather than enabled for
the measured stable configuration.

## Selected settings

- Context length: 2,048
- Micro-batch size: 2
- Gradient accumulation: 16
- Tokens per optimizer update: 65,536
- Gradient checkpointing: disabled initially
- Compilation: disabled initially

An actual AdamW update at the selected batch and context used less than half
of visible VRAM. This leaves practical space for allocator variation,
DataLoader transfers, validation, and checkpoint operations.

At the measured 22.8k tokens/second, uninterrupted raw compute would take
about 36.5 hours for 3B tokens or 60.9 hours for 5B tokens. Real runs will be
longer because validation, generation, checkpointing, startup, and downtime
are excluded.

## Compilation limitation

The default Inductor backend was attempted and failed because the Fedora host
does not currently provide `/usr/include/python3.14/Python.h`. No system
package was installed implicitly. The supported `aot_eager` backend completed
forward and backward successfully, but its compile overhead makes it unsuitable
for this run. Eager mode is therefore the approved baseline.

The ignored machine-readable benchmark is at
`logs/phase13/benchmark.json`. The benchmark is reproducible with
`python -m scripts.benchmark_250m`.
