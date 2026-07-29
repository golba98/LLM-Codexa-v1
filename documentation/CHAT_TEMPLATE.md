# Codexa Instruction Template

Instruction tuning is optional and produces a separate model from the base
pretrained checkpoint. The base checkpoint is never overwritten.

Chat template version `1.0` formats one turn as:

```text
<bos>User: {instruction}
Context: {optional context}
Assistant: {response}<eos>
```

The `Context:` line is omitted when no context is supplied. BOS and EOS are
inserted as token IDs; they are not embedded as ordinary text. During
supervised fine-tuning, labels for BOS, the user instruction, optional context,
the `Assistant:` prefix, and padding are set to `-100`. Loss is calculated only
for the assistant response and EOS.

Long prompts are deterministically truncated from their middle so both the
beginning of the instruction and the `Assistant:` suffix survive. Responses
are truncated only when the complete example would exceed the model's
2,048-token context.

The pinned instruction dataset is Databricks Dolly 15k revision
`bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, licensed CC-BY-SA 3.0. Its
share-alike and attribution terms apply to instruction-tuned derivatives and
must be reviewed before release.
