# Dataset Strategy and Provenance

## Repository fixtures

The tracked files under `tests/fixtures/data/` are original, small test
fixtures. The million-token Phase 11 corpus is a deterministic synthetic
expansion of that material and is used only to verify the pipeline.

## TinyStories

Phases 14 and 15 use
[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
a corpus of synthetically generated English short stories. The dataset card
declares the CDLA-Sharing-1.0 license.

The download is pinned to dataset revision:

`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`

`scripts/download_tinystories.py` downloads the official train and validation
text files and records byte sizes and SHA-256 checksums. Raw and processed
files remain ignored by Git.

TinyStories is useful for validating coherent generation at modest compute,
but it is not a broad general-knowledge or code corpus. Results from it must
not be represented as general language ability. Before Phase 16 begins, the
dataset decision must be revisited because the roadmap's 3B–5B-token target
and code-completion evaluation require broader coverage than TinyStories
provides.
