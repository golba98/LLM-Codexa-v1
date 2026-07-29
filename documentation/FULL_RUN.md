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

The first real-input preflight was recorded at
`2026-07-29T09:07:52Z`. It proved:

- NVIDIA GeForce RTX 4080 and CUDA available
- CUDA BF16 supported
- GPU temperature 65 C, below the configured 70 C maximum
- production token manifest and binaries valid
- 438 GiB available on the checkpoint filesystem
- conservative retained-checkpoint projection 146,207,896,576 bytes
  (about 136.2 GiB)

The run remains intentionally gated on:

- selecting and verifying an independently mounted backup destination;
- explicit confirmation of stable mains or UPS power, because this desktop
  exposes no Mains/UPS state through `/sys/class/power_supply`;
- the final isolated BF16 forward/backward preflight after the intermediate
  GPU run releases the device.

The full run must not start until every preflight check reports `pass`.

## Reproducible preflight

```bash
.venv/bin/python -m scripts.preflight_full_run \
  --config configs/250m_full.yaml \
  --token-manifest data/tokenized/fineweb-edu/token_data_manifest.json \
  --train-token-file data/tokenized/fineweb-edu/train.bin \
  --validation-token-file data/tokenized/fineweb-edu/validation.bin \
  --checkpoint-dir checkpoints/phase16-full \
  --backup-destination /path/on/independent/filesystem \
  --maximum-temperature 70 \
  --confirm-power-stability \
  --model-smoke \
  --output logs/phase16-full/preflight_final.json
```

`--confirm-power-stability` is an explicit human assertion and must only be
used after the power source has actually been checked.
