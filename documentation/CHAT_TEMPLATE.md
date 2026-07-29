# Codexa Chat Template

Chat instruction tuning produces a separate model from the base pretrained
checkpoint. The base checkpoint is never overwritten and the architecture
remains exactly 248,565,504 parameters.

Chat template version `2.0` formats a conversation as:

```text
<bos>System: {optional system instruction}
User: {first user message}
Assistant: {first assistant response}
User: {next user message}
Assistant: {next assistant response}<eos>
```

The system message is optional and may appear only first. User and assistant
roles must then alternate, training records must end with an assistant, and
inference prompts must end with a user. Legacy Dolly `context` is included
inside its user message as `Context: {context}`.

BOS and EOS are inserted as token IDs; they are not embedded as ordinary text.
During supervised fine-tuning, labels for BOS, role prefixes, system messages,
user messages, and padding are set to `-100`. Causally shifted loss is
calculated only for every assistant response and the final EOS.

At inference time, `scripts.generate --instruction` uses the same prefix and
prepends BOS token ID 1. It does not append EOS to the prompt; generation
stops when the model emits EOS or reaches the context limit.

For long multi-turn records, the oldest complete user/assistant pairs are
dropped first. If the final exchange alone remains too long, user content is
shortened from the left and assistant content from the right. Role prefixes,
the latest exchange, target alignment, and the 2,048-token limit are preserved.

The chat dataset is derived deterministically from Databricks Dolly 15k revision
`bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, licensed CC-BY-SA 3.0. Its
share-alike and attribution terms apply to instruction-tuned derivatives and
must be reviewed before release. Version 1 contains 14,996 conversations:
12,050 single-turn and 2,946 deterministic context-recall multi-turn records.
Its frozen SHA-256 is
`2775dbd9e83933c2a6631329df14ad722920e6a70c205f8d4a744e371c119fb8`.

The selected chat checkpoint is step 2,600 of `codexa-v1-chat-3epoch`, with
validation loss 2.4662499567521. It improves conversational formatting,
information extraction, summarization, and context recall over the TinyStories
base, but remains unreliable for factual answers and strict instruction
following.
