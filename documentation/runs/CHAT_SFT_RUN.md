# Codexa v1 Chat SFT Run

## Selected model

The chat run preserves the existing architecture exactly:

- Parameters: 248,565,504
- Layers: 34
- Hidden size: 768
- Attention heads: 12
- Context length: 2,048
- Vocabulary size: 8,192
- Base checkpoint: `checkpoints/phase15-500m/best.pt`
- Base optimizer step: 7,500

The base checkpoint remains unchanged. Chat checkpoints are stored separately
under `checkpoints/codexa-v1-chat/`.

## Dataset

`codexa-chat-v1` was prepared from the pinned Databricks Dolly 15k source:

- License: CC-BY-SA 3.0
- Records: 14,996
- Single-turn records: 12,050
- Multi-turn records: 2,946
- Messages: 35,884
- Chat template: 2.0
- Dataset SHA-256:
  `2775dbd9e83933c2a6631329df14ad722920e6a70c205f8d4a744e371c119fb8`

The multi-turn subset contains deterministic context-recall or exact-repeat
follow-ups. It does not add unsupported factual answers.

## Initial one-epoch run

The first CUDA BF16 run completed 1,000 optimizer steps. It established that
the pipeline was stable, but covered only about one pass over the training
split:

- Optimizer steps: 1,000
- Micro-steps: 8,000
- Tokens processed, including padding and repeated samples: 6,914,034
- Initial training loss: 5.323554
- Final training loss: 2.646678
- Best validation loss: 2.719971 at step 900
- Peak allocated VRAM: 10,503,376,896 bytes
- Peak reserved VRAM: 11,997,806,592 bytes

Its selected checkpoint was step 900 with validation loss 2.719971.

## Selected three-epoch run

`configs/250m_chat.yaml` defines 2,661 optimizer steps with an effective batch
of 16, for 42,576 examples—almost exactly three passes over the 14,190-record
training split. The clean run started from the preserved Phase 15 base rather
than continuing from the one-epoch SFT model.

- Optimizer steps: 2,661
- Micro-steps: 21,288
- Completed DataLoader epochs: 3
- Tokens processed, including padding: 18,403,306
- Initial training loss: 5.323554
- Final training loss: 2.184958
- Best validation loss: 2.466250 at step 2,600
- Peak allocated VRAM: 10,557,756,416 bytes
- Peak reserved VRAM: 11,997,806,592 bytes

Selected checkpoint:

```text
checkpoints/codexa-v1-chat-3epoch/best.pt
SHA-256: 5cc74f6e3f14ba6a57268054e8c90395b40ca5c98c10baad9479e139b912f191
```

Tokenizer:

```text
checkpoints/tokenizer-tinystories/tokenizer.json
SHA-256: 66df4e459b95af715b704c3a576f872db97e6e4e7e86d4774865c152d5221e98
```

## Evaluation

The fixed eight-prompt instruction suite was run against the base, 10-step
pilot, one-epoch checkpoint, and selected three-epoch checkpoint:

| Checkpoint | Expected terms | Strict formats |
| --- | ---: | ---: |
| Phase 15 base | 0/5 | 0/1 |
| One-epoch SFT | 0/5 | 1/1 |
| Three-epoch SFT | 1/5 | 0/1 |

The three-epoch model extracted `Orion` from the fixed prompt. Live API tests
also correctly recalled that the user's favorite color was blue, retained
`Project Orion` from a note, and summarized a bee sentence without continuing
as a story. These are measurable improvements, while the failed constraints
remain visible.

This is evidence of learned chat formatting and limited context recall, not
reliable general-assistant quality. The model still produces incorrect facts,
invented words, and weak constraint following. Its narrow TinyStories
pretraining is the main limitation; broader pretraining data and stronger
licensed instruction/chat data are required before Phase 18 can be considered
complete.

### Rejected category-balanced experiment

A controlled run used the same base checkpoint, dataset, seed, optimizer, and
2,661-step schedule while sampling the eight top-level Dolly categories with
equal probability. It was rejected:

- Best validation loss: 2.514371 at step 2,600
- Selected unbalanced validation loss: 2.466250 at step 2,600
- Both models matched one of five expected terms and zero of one strict format
- The balanced model lost the correct `Orion` extraction
- Its apparent temperature match was an accidental zero inside a malformed
  number

The selected checkpoint remains `codexa-v1-chat-3epoch/best.pt`. The
category-balancing implementation was removed rather than retaining a
negative experiment as unused product complexity.

Evaluation reports and training metrics are local ignored artifacts:

```text
logs/codexa-v1-chat/train_metrics.jsonl
logs/codexa-v1-chat/run_metadata.json
logs/phase18-chat-evaluation/base.json
logs/phase18-chat-evaluation/pilot.json
logs/phase18-chat-evaluation/final-best.json
logs/phase18-chat-evaluation/three-epoch-best.json
```

## LM Studio

The local OpenAI-compatible server uses the selected step-2,600 checkpoint,
CUDA BF16, and canonical multi-turn template. Its model ID is
`codexa-v1-chat`. Non-streaming and streaming API requests were verified.
The user-level service is:

```text
codexa-lmstudio.service
http://127.0.0.1:1235/v1
```
