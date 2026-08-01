# Codexa Chat Protocol

Chat template version `3.0` is a token-level protocol shared by dataset
encoding and OpenAI-compatible inference. It uses these fixed IDs:

| Token | ID |
| --- | ---: |
| `<pad>` | 0 |
| `<bos>` | 1 |
| `<eos>` | 2 |
| `<unk>` | 3 |
| `<|system|>` | 8192 |
| `<|user|>` | 8193 |
| `<|assistant|>` | 8194 |
| `<|end|>` | 8195 |

The vocabulary therefore contains 8,196 entries. For the tied 1B-class model,
the four new embedding rows increase the exact parameter count from
920,200,704 to 920,206,848.

An inference prompt is serialized as:

```text
<bos><|system|>
{system content}<|end|>
<|user|>
{user content}<|end|>
<|assistant|>
```

An absent system message becomes one empty system block. Each completed
assistant turn appends `{assistant content}<|end|>`. Structural IDs are added
manually; the tokenizer never adds another BOS, EOS, role, or end token.
Role-token-looking text inside message content is encoded as ordinary text.

SFT labels are `-100` for the prompt, structural headers, user/system content,
and padding. Every assistant response and its terminating `<|end|>` contribute
to loss. Conversations are not packed across record boundaries. Pretraining
continues to place one EOS between unrelated source documents.

Inference stops on EOS, `<|end|>`, or any newly generated role token. The
terminator is counted but never shown. `finish_reason` is `stop` for those
terminations and `length` only when the generation budget is exhausted.

The base pretraining checkpoint does not implement this protocol. It must be
fine-tuned with the extended tokenizer before the chat server will load it.
