# Base Tokenizer

Codexa uses Hugging Face `tokenizers` with byte-level BPE, a ByteLevel
pre-tokenizer, and a ByteLevel decoder. The vocabulary contains 8,192 entries.

| Token | ID |
| --- | ---: |
| `<pad>` | 0 |
| `<bos>` | 1 |
| `<eos>` | 2 |
| `<unk>` | 3 |

No BOS or EOS token is silently added to prompts. Dataset tokenization appends
exactly one EOS token after every document.

The active FineWeb-Edu tokenizer has SHA-256:

```text
6b26d3c98d8782298119875c368a69fdccbff03cca6fbfa1fc0851b0f3f8ef0c
```

It produced zero unknown tokens while inspecting the retained prepared shard.
It remains English-focused and is not specialized for multilingual text or
code. Any future vocabulary change creates a new base model and must not be
mixed with existing token binaries or checkpoints.
