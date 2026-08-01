# Tokenizer

Codexa uses Hugging Face `tokenizers` with a byte-level BPE model, ByteLevel
pre-tokenizer, and ByteLevel decoder. Special tokens have fixed IDs:

| Token | ID |
| --- | ---: |
| `<pad>` | 0 |
| `<bos>` | 1 |
| `<eos>` | 2 |
| `<unk>` | 3 |

Chat fine-tuning appends `<|system|>`, `<|user|>`, `<|assistant|>`, and
`<|end|>` at IDs 8192 through 8195 without changing the existing BPE rows.
Create the versioned chat tokenizer with:

```bash
.venv/bin/python -m scripts.extend_chat_tokenizer \
  --input checkpoints/tokenizer-fineweb-edu/tokenizer.json \
  --output-dir checkpoints/tokenizer-fineweb-edu-chat-v3
```

The tokenizer has no automatic BOS or EOS insertion. Dataset tokenization
appends exactly one EOS after each document.

## TinyStories tokenizer

The Phase 14–15 tokenizer was trained deterministically from the cleaned
TinyStories train and validation JSONL streams with a requested and actual
vocabulary size of 8,192. Measured on that prepared corpus:

- Documents: 1,814,637
- Characters: 1,605,283,279
- Tokens before document-separator EOS insertion: 386,143,439
- Average characters per token: 4.1572
- Unknown-token rate: 0.0
- Tokenizer SHA-256:
  `66df4e459b95af715b704c3a576f872db97e6e4e7e86d4774865c152d5221e98`

The tokenizer is suitable for the selected English story corpus. Its compact
vocabulary lowers embedding cost, but it has not been optimized for code,
technical notation, broad multilingual text, or chat templates.

## FineWeb-Edu tokenizer

The Phase 16 tokenizer was trained from the complete, frozen four-shard
FineWeb-Edu candidate using the streaming byte-level BPE path, then inspected
over every cleaned train and validation document. Requested and actual
vocabulary sizes are both 8,192.

Measured on the prepared corpus before document-separator EOS insertion:

- Documents: 2,882,129
- Characters: 13,677,423,520
- UTF-8 bytes: 13,746,602,145
- Content tokens: 3,511,159,117
- Average characters per token: 3.895415
- Unknown-token count and rate: 0 and 0.0
- Tokenizer SHA-256:
  `6b26d3c98d8782298119875c368a69fdccbff03cca6fbfa1fc0851b0f3f8ef0c`

Adding exactly one EOS token per document produces 3,514,041,246 stored
tokens across both splits:

| Split | Documents | Tokens | Complete 2,048-token sequences | Trailing tokens |
| --- | ---: | ---: | ---: | ---: |
| Train | 2,867,754 | 3,496,587,568 | 1,707,318 | 303 |
| Validation | 14,375 | 17,453,678 | 8,522 | 621 |

The compact binary dtype is `uint16`. Binary SHA-256 checksums are:

- Train:
  `4575f7e3b837909ea512433d85c453a686466a74a14d2bfcfcaeef39728fe9f3`
- Validation:
  `f621153329cc54195986d47302e8f541d782f75e3ac971fa05d20e3b0295ef75`

Full checksum, token-range, document-index contiguity, and EOS-position
validation passed with `scripts.inspect_token_data`.

FineWeb-Edu broadens educational and general web coverage relative to
TinyStories. It remains English-focused, is not a specialized code corpus,
and does not make the base model instruction-tuned.
