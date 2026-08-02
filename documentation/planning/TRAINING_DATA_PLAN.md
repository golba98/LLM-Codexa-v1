# Codexa v1 Training Data and Conversation Plan

## Short answer: will it talk better?

Yes, this rebuild gives Codexa a much better route to usable conversation, but
base training alone does not create a finished chat assistant.

The two training stages teach different capabilities:

1. **Base pretraining** on FineWeb-Edu and Wikipedia teaches English, general
   information, factual-style prose, topic continuity, and text completion.
2. **Conversation fine-tuning** on UltraChat and OASST1 teaches user/assistant
   turns, direct answers, follow-up responses, and when an assistant reply ends.

If we stop after base pretraining, the model should complete text more
coherently, but it may answer prompts like an article rather than a helpful
assistant. If the base model is weak, conversational fine-tuning will not fix
the missing language foundation. Both stages are required.

Training also cannot guarantee truthfulness, reasoning ability, or ChatGPT-level
quality. Codexa is a local 921,773,568-parameter research model, so acceptance
depends on measured output rather than the training run merely finishing.

## The downloaded datasets

The raw downloads are complete and stored under the ignored `data/raw/`
directory. They are source data, not yet a final tokenized training mixture.

| Stage | Dataset | Downloaded content | Local size | Records | What it contributes |
| --- | --- | --- | ---: | ---: | --- |
| Base | FineWeb-Edu | 10 `sample-10BT` Parquet shards | 21.5 GB | 7,293,000 documents | Broad English, educational explanations, and general web knowledge |
| Base | English Wikipedia | Complete `20231101.en` snapshot | 11.6 GB | 6,407,814 articles | Encyclopedic topics, names, concepts, history, science, and factual-style prose |
| Chat SFT | UltraChat 200k | `train_sft` and `test_sft` | 813 MB | 230,975 conversations | Instruction-and-response patterns across many topics |
| Chat SFT | OASST1 | Train and validation | 42 MB | 88,838 message rows | Human conversation trees, roles, follow-ups, and quality metadata |

Every download is pinned to a repository revision. Each local corpus has a
`download_manifest.json` containing selected files, byte sizes, and SHA-256
checksums. Exact revisions and licenses are recorded in
`documentation/reference/DATASET.md`.

## What must happen before base training

The raw data cannot be passed straight into `scripts/train.py`. The preparation
pipeline must first produce one reproducible base mixture.

1. Clean and normalize FineWeb-Edu and Wikipedia independently.
2. Remove empty, malformed, and exact duplicate documents.
3. Create deterministic train and validation splits without leaking documents
   between them.
4. Measure each source in documents and tokens.
5. Choose and record the FineWeb-Edu/Wikipedia token mixture. Do not repeatedly
   oversample Wikipedia until it dominates the general corpus.
6. Train or freeze the 8,192-entry tokenizer against the approved base data.
7. Convert the prepared text into memory-mapped token files.
8. Write a manifest containing input checksums, tokenizer checksum, mixture,
   split counts, token counts, seed, and code revision.

The existing 883,814,184-token artifact represents only the original first
FineWeb-Edu shard. It remains useful for smoke testing, but it is not the new
FineWeb-Edu plus Wikipedia production corpus.

## Stage A: validate the 1B training path

All validation uses the actual 921,773,568-parameter configuration. There is no
250M prerequisite.

Before a long run:

- run the full test suite;
- prove the 1B model can overfit the tiny deterministic fixture;
- run a short base-mixture smoke job from random weights;
- verify causal labels, document boundaries, validation separation, checkpoint
  recovery, tokenizer compatibility, finite loss, and falling loss;
- compare fixed prompt output before and after the smoke run.

A failed check stops the launch. Sampling settings must not be used to disguise
bad data, broken labels, repetition, or a weak checkpoint.

## Stage B: train the 1B base model

Start the model from new random weights and train it on only the approved
FineWeb-Edu plus Wikipedia token stream. Record processed tokens, not only
optimizer steps.

At milestones, save checkpoints and run the same fixed prompts covering:

- ordinary English prose;
- educational explanations;
- factual-style and encyclopedic continuations;
- short stories;
- headings, lists, quotations, and paragraphs;
- repetition, premature stopping, and malformed token output.

The base passes only if unseen prompts produce readable, connected text across
multiple topics and held-out loss improves. This checkpoint is preserved before
any conversational training begins.

## Stage C: prepare conversation data

UltraChat and OASST1 must remain separate from the base corpus. Flattening them
into plain text would throw away who said each message and which tokens should
teach the assistant.

Preparation must:

1. preserve `user` and `assistant` roles;
2. reconstruct valid OASST1 conversation paths from its message trees;
3. remove deleted, malformed, unsafe, duplicate, and low-quality examples;
4. retain genuine multi-turn examples, not only single question/answer pairs;
5. serialize both datasets with one versioned chat format;
6. calculate loss on assistant answers only;
7. create conversation-level train and validation splits so turns from one
   conversation cannot leak across splits;
8. record the final source mixture and supervised-token count.

UltraChat provides breadth but was model-generated and filtered. OASST1 adds
human-generated and human-annotated conversations. Their quality must be
measured before choosing the final mixture.

## Stage D: conversation fine-tuning

Fine-tune a copy of the accepted base checkpoint on the prepared conversational
mixture. Do not overwrite the base checkpoint.

The resulting model must pass fixed multi-turn tests for:

- answering the user's actual question directly;
- remembering a name or fact introduced earlier in the conversation;
- handling a correction in a later turn;
- following simple formatting instructions;
- producing one assistant response and stopping correctly;
- avoiding repeated phrases and conversation loops;
- retaining the base model's general-language ability.

Only a checkpoint that passes these tests should be exposed through a chat
interface. A low training loss by itself is not proof that it can converse.

## Current status

- [x] Select existing licensed general, Wikipedia, and conversation datasets.
- [x] Download and checksum 10 FineWeb-Edu shards.
- [x] Download and checksum the complete English Wikipedia snapshot.
- [x] Download and checksum UltraChat SFT and OASST1.
- [x] Validate Parquet schemas, record counts, manifests, and file sizes.
- [ ] Implement the deterministic FineWeb-Edu plus Wikipedia preparation path.
- [ ] Decide the base token mixture after measuring cleaned token counts.
- [ ] Build and verify the new tokenizer and token stream.
- [ ] Pass the 1B tiny-overfit and base-mixture smoke gates.
- [ ] Train and select the 1B base checkpoint.
- [ ] Build the role-aware conversational SFT mixture.
- [ ] Fine-tune and evaluate the conversational checkpoint.

The immediate next task is data preparation and token accounting. Starting the
long training run before that work is complete would train on the wrong or
incompletely prepared input.
