# Base Dataset Strategy and Provenance

Codexa's base rebuild uses existing licensed data. The project does not create
or hand-author a pretraining corpus.

## Dataset roles

| Purpose | Dataset | Selected content | License |
| --- | --- | --- | --- |
| General/educational base text | `HuggingFaceFW/fineweb-edu` | ten leading `sample-10BT` shards | ODC-By 1.0 |
| Encyclopedic base text | `wikimedia/wikipedia` | complete `20231101.en` snapshot | CC BY-SA 3.0 and GFDL |
| Later conversational SFT | `HuggingFaceH4/ultrachat_200k` | `train_sft`, `test_sft` | MIT |
| Later conversational SFT | `OpenAssistant/oasst1` | train, validation | Apache-2.0 |

The first two sources are base-pretraining candidates. The conversational
sources are not generic prose: preserve message roles and apply loss only to
the intended assistant targets during the later SFT stage.

## FineWeb-Edu

The active source is `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`,
distributed under ODC-By 1.0 and pinned to revision:

```text
87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
```

The retained local artifact currently comes from one upstream Parquet shard:

```text
data/raw/fineweb-edu-10bt/sample/10BT/000_00000.parquet
```

Its SHA-256 is:

```text
b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871
```

Preparation retained 724,510 clean unique documents from 726,000 rows and
removed 1,490 exact duplicates. The seed-42 split contains 720,903 training
documents and 3,607 validation documents.

The active token artifact contains:

| Split | Documents | Stored tokens | Complete 2,048-token sequences |
| --- | ---: | ---: | ---: |
| Train | 720,903 | 883,814,184 | 431,549 |
| Validation | 3,607 | 4,453,653 | 2,174 |

The tokenizer SHA-256 is:

```text
6b26d3c98d8782298119875c368a69fdccbff03cca6fbfa1fc0851b0f3f8ef0c
```

The retained shard verifies the complete local pipeline but is not the final
training corpus for a 1B-class model. Additional pinned FineWeb-Edu shards must
be prepared and tokenized before the production run. Exact input files, output
checksums, document counts, and token counts must be frozen in manifests.

FineWeb-Edu is filtered web text. It may contain factual errors, bias, private
information, unsafe material, copyrighted text, and duplication missed by
exact hashing. It is English-focused and is not a specialized code or dialogue
corpus.

## Wikipedia

The complete English `20231101.en` Parquet snapshot is pinned to revision
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`. It supplies sustained
encyclopedic prose and broad topic coverage. Wikipedia is not a guarantee of
factual correctness, and its share of the final token mixture must be measured
rather than allowed to dominate through repeated epochs.

## Conversational sources

UltraChat 200k is pinned to revision
`8049631c405ae6576f93f445c6b8166f76f5505a`; only its SFT train and test
splits are downloaded. Its conversations were model-generated and filtered,
so it is useful for broad turn formats but must be quality-audited.

OASST1 is pinned to revision
`fdf72ae0827c1cda404aff25b6603abec9e3399b`. It contains human-generated and
human-annotated conversation trees. Tree relationships, roles, and quality
metadata must be preserved when selecting SFT paths.

Each raw directory contains a `download_manifest.json` with the repository,
revision, selected files, byte sizes, and SHA-256 checksums. Raw data is local
and ignored by Git.

## Repository fixtures

Small original files under `tests/fixtures/data/` exist only for deterministic
pipeline and overfit tests. They are not production training data.
