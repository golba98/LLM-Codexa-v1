# Phase 22 — Evaluation and Release

## Goal

Choose a checkpoint based on evidence and make it usable through LM Studio.

## Validation

- Compare the 250M baseline and 1B candidates on fixed prompts.
- Measure validation loss, coherence, repetition, instruction following,
  context retention, malformed output, and factual limitations.
- Verify checkpoint and tokenizer checksums.
- Confirm `/v1/models` and chat completions through the local API.

## LM Studio

Start the local server with the selected 1B chat checkpoint, verify port 1235,
then select `golba98/codexa-openai-adapter` in LM Studio. Test the same fixed
prompts through the API and UI.

## Release contents

- Separate base and chat checkpoints
- Tokenizer, checksums, configuration, and dataset manifests
- Evaluation report and known limitations
- Updated model card and reproducible run command
