# Phase 21 — Conversation SFT

## Goal

Teach the 1B base model to follow the canonical `User:` / `Assistant:` format,
retain recent context, and answer naturally across multiple turns.

## Starting data

```text
data/processed/codexa-chat-conv-v2/chat.jsonl
```

The set contains cleaned instruction examples plus quality-filtered OASST1
multi-turn chains. Recheck placeholders, role order, duplicates, truncation,
and license metadata before every run.

## Run policy

1. Run a small overfit/pilot test to verify assistant-only loss masking.
2. Train for 2–3 epochs with a held-out validation split.
3. Test greetings, identity, follow-ups, corrections, repetition, and context.
4. Keep the 1B base checkpoint separate from chat checkpoints.

## Acceptance criteria

- Responses are non-empty and use the correct assistant role.
- Known placeholder templates are not reproduced.
- Earlier relevant information is retained across turns.
- Repetition and malformed-output rates are documented.
