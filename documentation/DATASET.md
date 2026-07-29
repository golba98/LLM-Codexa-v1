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
project near 4.5B tokens before exact cross-shard deduplication.

The four-shard candidate was downloaded and prepared from these exact upstream
files:

| Shard | Bytes | SHA-256 |
| --- | ---: | --- |
| `000_00000.parquet` | 2,152,819,114 | `b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871` |
| `001_00000.parquet` | 2,152,222,432 | `3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693` |
| `002_00000.parquet` | 2,151,796,315 | `547ae182d132c9f06b6ce63149567208ea9f57630bfd9b1a2938e504f0c9ebd7` |
| `003_00000.parquet` | 2,152,437,524 | `22184e6eb25759ddd97783751ffc73e1705dfa2542e630dae1f2a8bac8ee6ddb` |

Preparation read 2,916,000 raw rows, retained 2,882,129 clean unique
documents, and removed 33,871 exact duplicates. The seed-42 split contains
2,867,754 training and 14,375 validation documents (0.5% validation), with
13,677,423,520 total characters. The clean output checksums are:

- Train:
  `fcd049d7a744b3e3c32186a10fd97a59b304c1edd801c4bb471c22b9dd5d5e37`
- Validation:
  `b39bb8803ac9f32fd03907f33b9cf66336a9aa6b55c4d6419f71f83fdc9fcd7a`

The broad-corpus tokenizer inspection covered all 2,882,129 documents and
measured 3,511,159,117 content tokens with zero unknown tokens. Adding one EOS
per document gives an exact projected stored total of 3,514,041,246 tokens,
inside the roadmap's 3B–5B target. The tokenizer vocabulary is 8,192 and its
SHA-256 is:

`6b26d3c98d8782298119875c368a69fdccbff03cca6fbfa1fc0851b0f3f8ef0c`

The cleaned JSONL inputs, tokenizer, and token binaries are frozen. The atomic
binary build produced:

| Split | Documents | Tokens | Bytes | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Train | 2,867,754 | 3,496,587,568 | 6,993,175,136 | `4575f7e3b837909ea512433d85c453a686466a74a14d2bfcfcaeef39728fe9f3` |
| Validation | 14,375 | 17,453,678 | 34,907,356 | `f621153329cc54195986d47302e8f541d782f75e3ac971fa05d20e3b0295ef75` |

At context length 2,048, strict non-overlapping packing exposes 1,707,318
complete training sequences and 8,522 validation sequences, dropping only 303
and 621 trailing tokens respectively. Full checksum, token-range,
document-index contiguity, and EOS-position validation passed.

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
