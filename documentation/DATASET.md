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

## FineWeb-Edu

Phase 16 uses a measured subset of the official
[HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
`sample-10BT` configuration. FineWeb-Edu is educational web text filtered from
FineWeb and is distributed under ODC-By 1.0. The source must be attributed in
any dataset or model release.

Downloads are pinned to revision:

`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`

`scripts/download_fineweb_edu.py` downloads a deterministic leading shard set
and records SHA-256 checksums. `scripts/prepare_fineweb_edu.py` streams Parquet
row groups, applies the same cleaning policy as the local pipeline, performs
exact text deduplication through an on-disk SQLite index, and creates a stable
hash-based validation split. The number of shards used for the final run is
selected only after tokenization confirms that the cleaned corpus contains the
roadmap's 3B–5B Codexa-token target.

The initial sizing sample encoded 10,000 cleaned documents from shard 0 into
15,362,976 stored tokens (including one EOS per document), or 3.0815
characters per content token. With 726,000 rows in that shard, four shards
project near 4.5B tokens before exact cross-shard deduplication. Four is the
provisional selection; the dataset is not frozen until full preparation and
tokenization produce an exact count within 3B–5B.

FineWeb-Edu broadens subject coverage, but web filtering is imperfect. It can
contain factual errors, bias, personally identifying text, unsafe material,
copyrighted text, and duplication missed by exact matching. It is not a
specialized code corpus, so code-completion results must be reported honestly
rather than treated as a primary capability.

## Databricks Dolly 15k

Optional supervised fine-tuning uses
[databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
pinned to revision:

`bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`

The dataset contains 15,011 human-authored instruction/response records across
brainstorming, classification, closed and open question answering, creative
writing, general question answering, information extraction, and
summarization. It is licensed CC-BY-SA 3.0. The downloaded JSONL checksum is:

`2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`

`scripts/download_dolly.py` records the pinned source files, byte sizes,
license, and SHA-256 checksums. The deterministic seed-42 split contains
14,221 training and 790 validation records. Dolly's attribution and share-alike
terms apply to instruction-tuned derivatives; the base pretrained checkpoint
remains separate.
