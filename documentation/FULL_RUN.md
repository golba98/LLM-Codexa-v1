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

## Live run

The full run started at `2026-07-29T13:22:51Z` with run ID
`876ad62d-631a-4eb5-aa41-8ad8eea0b68a` and Git commit
`4001715201e63e7336331495d50adc6fee43a18a`. The persistent user services are:

- `codexa-phase16-training.service`
- `codexa-phase16-monitor.service`

Training logs are under `logs/phase16-pretrain/`, GPU telemetry is
`logs/phase16-full/gpu_metrics.jsonl`, and checkpoints are under
`checkpoints/phase16-full/phase16-pretrain/`. The trainer is wrapped in a
sleep/shutdown inhibitor, uses `SIGINT` for graceful service stops, and user
lingering is enabled so a desktop logout does not terminate the run. A reboot
still requires verified resume from `latest.pt`.

The initial production-shape observation completed 16 optimizer updates and
1,048,576 tokens. Loss moved from 9.1690 to 8.9157 at approximately 28,000
tokens/second. Peak reserved VRAM was 11,406,409,728 bytes; sustained telemetry
showed 99% utilization, 65 C, and approximately 250 W. The selected chat
inference service was stopped to release its GPU allocation, but its checkpoint
remains preserved.

The first scheduled validation completed at optimizer update 500 with loss
`5.9895875453948975` over the configured four validation batches. Training loss
at update 500 was `5.981600314378738`. The resulting `best.pt` and `latest.pt`
files are each 2,983,115,203 bytes, and both SHA-256 sidecars passed independent
`sha256sum -c` verification while training continued.

At update 1,000, training loss was `5.185806065797806` and validation loss
improved to `5.21401834487915`. The first numbered milestone,
`milestones/step_000001000.pt`, was written at 2,983,115,203 bytes with SHA-256
`d3d54da6a585a1aeca17194beee47e03af84d70670212a4365564cbcf9b901e5`.
The `best.pt`, `latest.pt`, `previous.pt`, and numbered milestone sidecars all
passed `sha256sum -c`. The checkpoint directory then occupied 12 GiB and the
primary filesystem retained 335 GiB free.

At update 1,500, training loss was `4.571388214826584` and validation loss
improved again to `4.557276368141174`. The refreshed `best.pt` SHA-256 is
`0adb087af641652c6fca5a71200857aa3ffbfe0c286d5757dac835902d674760`.
The new `best.pt`, `latest.pt`, and rotated `previous.pt` sidecars all passed
`sha256sum -c` while training continued.

At update 2,000, training loss was `4.099617287516594` and validation loss
improved to `4.024853229522705`. The second numbered milestone,
`milestones/step_000002000.pt`, has SHA-256
`5ef2080c32996805f42508660c9cf013c9e747d015be3a93c94d22669eb73b6b`.
All retained checkpoint sidecars passed `sha256sum -c`.

Learning-rate metrics also proved the zero-based scheduler boundary: update
2,000 used scheduler index 1,999 and learning rate `0.00029985`; update 2,001
used index 2,000 and reached exactly `0.0003`; update 2,002 began cosine decay.
The checkpoint directory then occupied 14 GiB with 333 GiB free on the primary
filesystem.

At update 2,500, training loss was `3.69601309299469` and validation loss
improved to `3.7033477425575256`. The refreshed `best.pt` SHA-256 is
`456a35d55708ed6679b9d8be5c5dd84385a300a1143a9c109334f86fac719c03`.
The `best.pt`, `latest.pt`, and rotated `previous.pt` sidecars all passed
`sha256sum -c`. Training remained healthy immediately afterward at update
2,559, with finite loss and gradient norm, approximately 29,670 tokens/second,
99% GPU utilization, 66 C, and 333 GiB free on the primary filesystem.

At update 3,000, training loss was `3.501509800553322` and validation loss
improved to `3.5382099747657776`. The third numbered milestone,
`milestones/step_000003000.pt`, has SHA-256
`c7b62529542df656df2ecd1c4f4dacebef00bca0d4df1dd9f535532e3a0cd862`;
the refreshed `best.pt` has the same digest. The `best.pt`, `latest.pt`,
`previous.pt`, and milestone files are each 2,983,115,203 bytes, and every
sidecar passed `sha256sum -c`. The checkpoint directory then occupied 17 GiB,
with 330 GiB free on the primary filesystem. Training and telemetry services
remained active at 99% GPU utilization and 66 C.

At update 3,500, training loss was `3.4376363158226013` and validation loss
improved to `3.4302520751953125`. The refreshed `best.pt` SHA-256 is
`2990362a39fd2cd6ca6e4155357dc0fa7abb4e5e649ffeeb420be1b42c03d852`.
The 2,983,115,203-byte `best.pt`, `latest.pt`, and rotated `previous.pt`
sidecars all passed `sha256sum -c`. The checkpoint directory remained 17 GiB,
the primary filesystem retained 330 GiB free, and both services remained
active at 99% GPU utilization and 65 C.

At update 4,000, training loss was `3.356853574514389` and validation loss
improved to `3.328778922557831`. The fourth numbered milestone,
`milestones/step_000004000.pt`, and refreshed `best.pt` both have SHA-256
`1d1b5546ebff0572f5782b2eeabf2979ba30732a7d43f5e824a55679693f5c56`.
The 2,983,115,203-byte `best.pt`, `latest.pt`, `previous.pt`, and numbered
milestone sidecars all passed `sha256sum -c`. The checkpoint directory then
occupied 20 GiB with 327 GiB free on the primary filesystem. Both services
remained active at 100% GPU utilization and 64 C.

At update 4,500, training loss was `3.271818220615387` and validation loss
improved to `3.275608718395233`. The refreshed `best.pt` SHA-256 is
`5e066a1d1c76a057f24554f3e575e2a7221628c938eb737978891ac64d70bc4d`.
The 2,983,115,203-byte `best.pt`, `latest.pt`, and rotated `previous.pt`
sidecars all passed `sha256sum -c`. The checkpoint directory remained 20 GiB,
the primary filesystem retained 327 GiB free, and both services remained
active at 100% GPU utilization and 64 C.

At update 5,000, training loss was `3.206748366355896` and validation loss
improved to `3.2105628848075867`. The fifth numbered milestone,
`milestones/step_000005000.pt`, and refreshed `best.pt` both have SHA-256
`a91245cead16da47957d1f9e8add103c0f87070326147d9669546932ac7dd2de`.
The 2,983,115,203-byte `best.pt`, `latest.pt`, `previous.pt`, and numbered
milestone sidecars all passed `sha256sum -c`. The checkpoint directory then
occupied 23 GiB with 324 GiB free on the primary filesystem. Both services
remained active at 99% GPU utilization and 65 C.

At update 5,500, training loss was `3.267526462674141` and validation loss
improved to `3.177953839302063`. The refreshed `best.pt` SHA-256 is
`9bb5c8d6f2f935c114090b85833cf6113749373f8e5147ece174f8a97f41a846`.
The 2,983,115,203-byte `best.pt`, `latest.pt`, and rotated `previous.pt`
sidecars all passed `sha256sum -c`. The checkpoint directory remained 23 GiB,
the primary filesystem retained 324 GiB free, and both services remained
active at 99% GPU utilization and 65 C.
