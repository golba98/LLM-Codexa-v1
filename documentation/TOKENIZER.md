# Tokenizer

Codexa uses Hugging Face `tokenizers` with a byte-level BPE model, ByteLevel
pre-tokenizer, and ByteLevel decoder. Special tokens have fixed IDs:

| Token | ID |
| --- | ---: |
| `<pad>` | 0 |
| `<bos>` | 1 |
| `<eos>` | 2 |
| `<unk>` | 3 |

The production TinyStories tokenizer was trained deterministically from the
cleaned train and validation JSONL streams with a requested and actual
vocabulary size of 8,192. It has no automatic BOS or EOS insertion. Dataset
tokenization appends exactly one EOS after each document.

Measured on the prepared corpus:

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
