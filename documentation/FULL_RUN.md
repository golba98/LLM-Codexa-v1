# Full Pretraining Run

## Frozen inputs

Phase 16 uses these immutable inputs:

- Configuration: `configs/250m_full.yaml`
- Configuration SHA-256:
  `8d8ce9a00bb5bb6cffeae977c0c19d91aa804ddfe26a8778ea149277fc631841`
- Clean corpus: four pinned FineWeb-Edu `sample-10BT` shards documented in
  `documentation/DATASET.md`
- Tokenizer vocabulary: 8,192
- Tokenizer SHA-256:
  `6b26d3c98d8782298119875c368a69fdccbff03cca6fbfa1fc0851b0f3f8ef0c`
- Tokenizer manifest SHA-256:
  `a19e85e7230a6be6e5c82093949ac6990d4aa7e5866701a2a9e65a178fd77c39`
- Train tokens: 3,496,587,568
- Validation tokens: 17,453,678
- Train binary SHA-256:
  `4575f7e3b837909ea512433d85c453a686466a74a14d2bfcfcaeef39728fe9f3`
- Validation binary SHA-256:
  `f621153329cc54195986d47302e8f541d782f75e3ac971fa05d20e3b0295ef75`
- Token-data manifest SHA-256:
  `181bffa3266eaf308c84792b8770e447faf11105f06b9bd813defed218851fff`

The binary and index verifier passed checksum, dtype, vocabulary-range,
document-contiguity, and EOS-position checks.

## Frozen training target

The model has 248,565,504 parameters and a 2,048-token context. Training uses
BF16, micro-batch size 2, 16 accumulation steps, AdamW, peak learning rate
`3e-4`, 2,000 warmup steps, and 45,777 optimizer updates. The exact training
target is:

```text
45,777 updates x 16 micro-batches x 2 examples x 2,048 tokens
= 3,000,041,472 tokens
```

This target does not require cycling through the 3.497B-token training split.
Validation runs every 500 optimizer steps and milestone checkpoints are saved
every 1,000 steps.

## Preflight status

The latest real-input preflight was recorded at
`2026-07-29T12:30:48Z`. It proved:

- NVIDIA GeForce RTX 4080 and CUDA available
- CUDA BF16 supported
- isolated BF16 forward/backward passed at sequence length 512
- GPU temperature 45 C, below the configured 70 C maximum
- production token manifest and binaries valid
- 373.8 GiB available on the checkpoint filesystem
- conservative retained-checkpoint projection 146,207,896,576 bytes
  (about 136.2 GiB)

The original preflight was intentionally gated on an independently mounted
backup destination and explicit confirmation of stable mains or UPS power.
On 2026-07-29, the operator explicitly accepted running without an independent
checkpoint backup on the Fedora disk and confirmed stable power. This exception
must be recorded with `--accept-no-independent-backup`; omitting both that flag
and `--backup-destination` still fails closed.

The only other non-MSI physical disk discovered locally was inspected
read-only and unmounted again. It is a 512 GB SATA SSD labeled `Games`, with
49 GB free, so it cannot hold the 136.2 GiB retained-checkpoint projection and
is not a valid backup target. No files were written to it.

The current report is stored locally at:

```text
logs/phase16-full/preflight_current.json
```

All checks except `power_source` and `independent_backup` passed in that report.
The final report must record the operator's power confirmation and explicit
backup-risk acceptance.

The full run must not start until a final combined preflight reruns the model
smoke and every check reports `pass`.

## Reproducible preflight

```bash
.venv/bin/python -m scripts.preflight_full_run \
  --config configs/250m_full.yaml \
  --token-manifest data/tokenized/fineweb-edu/token_data_manifest.json \
  --train-token-file data/tokenized/fineweb-edu/train.bin \
  --validation-token-file data/tokenized/fineweb-edu/validation.bin \
  --checkpoint-dir checkpoints/phase16-full \
  --accept-no-independent-backup \
  --maximum-temperature 70 \
  --confirm-power-stability \
  --model-smoke \
  --output logs/phase16-full/preflight_final.json
```

`--confirm-power-stability` is an explicit human assertion and must only be
used after the power source has actually been checked.

`--accept-no-independent-backup` records an explicit risk decision; it cannot
be combined with `--backup-destination`. Checkpoints remain checksummed and
atomic, but a failure of the primary Fedora disk could still destroy them.

## Final preflight

The final preflight passed all nine checks at `2026-07-29T13:21:17Z` and is
stored at `logs/phase16-full/preflight_final.json`. It recorded:

- 248,565,504 model parameters;
- 365,605,363,712 free bytes against a 146,207,896,576-byte projection;
- explicit stable-power confirmation;
- explicit acceptance of running without an independent backup;
- RTX 4080 CUDA and BF16 support at 46 C;
- valid `uint16` production token data;
- a successful BF16 forward/backward pass at sequence length 512.

The no-backup waiver protects neither checkpoints nor logs from failure of the
primary Fedora disk. Atomic writes and SHA-256 sidecars only detect corruption;
they are not a substitute for an independent copy.
