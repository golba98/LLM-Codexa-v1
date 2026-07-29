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

## Training result

The CUDA BF16 run completed on the RTX 4080:

- Optimizer steps: 1,000
- Micro-steps: 8,000
- Tokens processed, including padding and repeated samples: 6,914,034
- Initial training loss: 5.323554
- Final training loss: 2.646678
- Best validation loss: 2.719971 at step 900
- Peak allocated VRAM: 10,503,376,896 bytes
- Peak reserved VRAM: 11,997,806,592 bytes

Selected checkpoint:

```text
checkpoints/codexa-v1-chat/best.pt
SHA-256: fcc6b1b44f808b05c3f9c26b0ffee2340ccc59b7afab47caf0ac9319e62af71f
```

Tokenizer:

```text
checkpoints/tokenizer-tinystories/tokenizer.json
SHA-256: 66df4e459b95af715b704c3a576f872db97e6e4e7e86d4774865c152d5221e98
```

## Evaluation

The fixed eight-prompt instruction suite was run against the base, 10-step
pilot, and selected step-900 checkpoint. The base passed zero strict format
patterns; the selected checkpoint passed one. A real multi-turn API test
correctly recalled that the user's favorite color was blue.

This is evidence of learned chat formatting and limited context recall, not
reliable general-assistant quality. The model still produces incorrect facts,
invented words, and weak constraint following. Its narrow TinyStories
pretraining is the main limitation; broader pretraining data and stronger
licensed instruction/chat data are required before Phase 18 can be considered
complete.

Evaluation reports and training metrics are local ignored artifacts:

```text
logs/codexa-v1-chat/train_metrics.jsonl
logs/codexa-v1-chat/run_metadata.json
logs/phase18-chat-evaluation/base.json
logs/phase18-chat-evaluation/pilot.json
logs/phase18-chat-evaluation/final-best.json
```

## LM Studio

The local OpenAI-compatible server uses the selected step-900 checkpoint,
CUDA BF16, and canonical multi-turn template. Its model ID is
`codexa-v1-chat`. Non-streaming and streaming API requests were verified.
The user-level service is:

```text
codexa-lmstudio.service
http://127.0.0.1:1235/v1
```
