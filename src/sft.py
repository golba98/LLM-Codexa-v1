"""Canonical chat formatting and response-masked fine-tuning datasets."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from src.chat_protocol import (
    CHAT_TEMPLATE_VERSION,
    ChatMessage,
    encode_chat_messages,
    format_chat_messages,
    validate_chat_messages,
    validate_chat_tokenizer,
)
from src.tokenizer import PAD_TOKEN


IGNORE_INDEX = -100


@dataclass(frozen=True)
class ChatRecord:
    """One complete supervised conversation ending with an assistant turn."""

    messages: tuple[ChatMessage, ...]
    category: str = "unknown"
    source: str = "unknown"
    conversation_id: str | None = None


@dataclass
class ChatLoadStatistics:
    """Deterministic accepted and removed-record counts for chat JSONL."""

    input_lines: int = 0
    accepted_records: int = 0
    malformed_json: int = 0
    invalid_schema: int = 0
    missing_or_invalid_roles: int = 0
    empty_responses: int = 0
    duplicate_examples: int = 0

    @property
    def removed_samples(self) -> int:
        return self.input_lines - self.accepted_records

    def to_dict(self) -> dict[str, int]:
        return {
            "input_lines": self.input_lines,
            "accepted_records": self.accepted_records,
            "malformed_json": self.malformed_json,
            "invalid_schema": self.invalid_schema,
            "missing_or_invalid_roles": self.missing_or_invalid_roles,
            "empty_responses": self.empty_responses,
            "duplicate_examples": self.duplicate_examples,
            "removed_samples": self.removed_samples,
        }


@dataclass(frozen=True)
class InstructionRecord:
    instruction: str
    context: str
    response: str
    category: str

    def to_chat_record(self) -> ChatRecord:
        """Convert the legacy Dolly-style record to the canonical chat schema."""

        user_content = self.instruction
        if self.context:
            user_content += f"\nContext: {self.context}"
        return ChatRecord(
            messages=(
                ChatMessage("user", user_content),
                ChatMessage("assistant", self.response),
            ),
            category=self.category,
            source="databricks-dolly-15k",
        )


def format_instruction_prompt(instruction: str, context: str = "") -> str:
    """Format a user instruction while leaving the response empty."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string.")
    if not isinstance(context, str):
        raise ValueError("context must be a string.")
    user_content = instruction.strip()
    if context.strip():
        user_content += f"\nContext: {context.strip()}"
    return format_chat_messages(
        (ChatMessage("user", user_content),),
        add_generation_prompt=True,
    )


def _parse_chat_messages(
    value: object,
    *,
    path: Path,
    line_number: int,
) -> tuple[ChatMessage, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"{path}:{line_number}: messages must be a non-empty array."
        )
    messages: list[ChatMessage] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(
                f"{path}:{line_number}: messages[{index}] must be an object."
            )
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str):
            raise ValueError(
                f"{path}:{line_number}: messages[{index}].role "
                "must be a string."
            )
        if not isinstance(content, str):
            raise ValueError(
                f"{path}:{line_number}: messages[{index}].content "
                "must be a string."
            )
        messages.append(ChatMessage(role=role, content=content.strip()))
    try:
        validate_chat_messages(messages, require_assistant_response=True)
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: {error}") from error
    return tuple(messages)


def _chat_record_from_value(
    value: object,
    *,
    path: Path,
    line_number: int,
) -> ChatRecord:
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: row must be an object.")
    if "messages" in value:
        messages = _parse_chat_messages(
            value["messages"],
            path=path,
            line_number=line_number,
        )
        category = value.get("category", "unknown")
        source = value.get("source", path.name)
        conversation_id = value.get("conversation_id")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"{path}:{line_number}: category must be non-empty.")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{path}:{line_number}: source must be non-empty.")
        if conversation_id is not None and not isinstance(conversation_id, str):
            raise ValueError(
                f"{path}:{line_number}: conversation_id must be a string or null."
            )
        return ChatRecord(
            messages=messages,
            category=category.strip(),
            source=source.strip(),
            conversation_id=conversation_id,
        )
    instruction = value.get("instruction")
    context = value.get("context", "")
    response = value.get("response")
    category = value.get("category", "unknown")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{path}:{line_number}: instruction must be non-empty.")
    if not isinstance(context, str):
        raise ValueError(f"{path}:{line_number}: context must be a string.")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"{path}:{line_number}: response must be non-empty.")
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"{path}:{line_number}: category must be non-empty.")
    return InstructionRecord(
        instruction=instruction.strip(),
        context=context.strip(),
        response=response.strip(),
        category=category.strip(),
    ).to_chat_record()


def load_chat_records_with_statistics(
    paths: Sequence[str | Path],
) -> tuple[list[ChatRecord], ChatLoadStatistics]:
    """Load all valid rows while accounting for every rejected sample."""

    statistics = ChatLoadStatistics()
    records: list[ChatRecord] = []
    seen: set[str] = set()
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                statistics.input_lines += 1
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    statistics.malformed_json += 1
                    continue
                try:
                    record = _chat_record_from_value(
                        value,
                        path=path,
                        line_number=line_number,
                    )
                except ValueError as error:
                    message = str(error).lower()
                    raw_messages = (
                        value.get("messages") if isinstance(value, dict) else None
                    )
                    has_empty_assistant = isinstance(raw_messages, list) and any(
                        isinstance(item, dict)
                        and item.get("role") == "assistant"
                        and (
                            not isinstance(item.get("content"), str)
                            or not item["content"].strip()
                        )
                        for item in raw_messages
                    )
                    if "response must be non-empty" in message or has_empty_assistant:
                        statistics.empty_responses += 1
                    elif "role" in message or "must end with assistant" in message:
                        statistics.missing_or_invalid_roles += 1
                    else:
                        statistics.invalid_schema += 1
                    continue
                identity = hashlib.sha256(
                    format_chat_messages(record.messages).encode("utf-8")
                ).hexdigest()
                if identity in seen:
                    statistics.duplicate_examples += 1
                    continue
                seen.add(identity)
                records.append(record)
                statistics.accepted_records += 1
    return records, statistics


def load_chat_records(path: str | Path) -> list[ChatRecord]:
    """Load canonical chat JSONL or legacy Dolly JSONL without skipping rows."""

    input_path = Path(path)
    records: list[ChatRecord] = []
    seen: set[str] = set()
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{input_path}:{line_number}: malformed JSON ({error.msg})"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"{input_path}:{line_number}: row must be an object."
                )
            if "messages" in value:
                messages = _parse_chat_messages(
                    value["messages"],
                    path=input_path,
                    line_number=line_number,
                )
                category = value.get("category", "unknown")
                source = value.get("source", input_path.name)
                conversation_id = value.get("conversation_id")
                if not isinstance(category, str) or not category.strip():
                    raise ValueError(
                        f"{input_path}:{line_number}: category must be "
                        "a non-empty string."
                    )
                if not isinstance(source, str) or not source.strip():
                    raise ValueError(
                        f"{input_path}:{line_number}: source must be "
                        "a non-empty string."
                    )
                if conversation_id is not None and not isinstance(
                    conversation_id, str
                ):
                    raise ValueError(
                        f"{input_path}:{line_number}: conversation_id must "
                        "be a string or null."
                    )
                record = ChatRecord(
                    messages=messages,
                    category=category.strip(),
                    source=source.strip(),
                    conversation_id=conversation_id,
                )
            else:
                instruction = value.get("instruction")
                context = value.get("context", "")
                response = value.get("response")
                category = value.get("category", "unknown")
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError(
                        f"{input_path}:{line_number}: instruction must "
                        "be non-empty."
                    )
                if not isinstance(context, str):
                    raise ValueError(
                        f"{input_path}:{line_number}: context must be a string."
                    )
                if not isinstance(response, str) or not response.strip():
                    raise ValueError(
                        f"{input_path}:{line_number}: response must be non-empty."
                    )
                if not isinstance(category, str) or not category.strip():
                    raise ValueError(
                        f"{input_path}:{line_number}: category must be non-empty."
                    )
                record = InstructionRecord(
                    instruction=instruction.strip(),
                    context=context.strip(),
                    response=response.strip(),
                    category=category.strip(),
                ).to_chat_record()
            identity = hashlib.sha256(
                format_chat_messages(record.messages).encode("utf-8")
            ).hexdigest()
            if identity in seen:
                continue
            seen.add(identity)
            records.append(record)
    if not records:
        raise ValueError("Chat dataset must not be empty.")
    return records


def load_instruction_records(path: str | Path) -> list[InstructionRecord]:
    """Load Dolly-style JSONL without silently skipping invalid rows."""

    input_path = Path(path)
    records: list[InstructionRecord] = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{input_path}:{line_number}: malformed JSON ({error.msg})"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"{input_path}:{line_number}: row must be an object."
                )
            instruction = value.get("instruction")
            context = value.get("context", "")
            response = value.get("response")
            category = value.get("category", "unknown")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(
                    f"{input_path}:{line_number}: instruction must be non-empty."
                )
            if not isinstance(context, str):
                raise ValueError(
                    f"{input_path}:{line_number}: context must be a string."
                )
            if not isinstance(response, str) or not response.strip():
                raise ValueError(
                    f"{input_path}:{line_number}: response must be non-empty."
                )
            if not isinstance(category, str) or not category.strip():
                raise ValueError(
                    f"{input_path}:{line_number}: category must be non-empty."
                )
            records.append(
                InstructionRecord(
                    instruction=instruction.strip(),
                    context=context.strip(),
                    response=response.strip(),
                    category=category.strip(),
                )
            )
    if not records:
        raise ValueError("Instruction dataset must not be empty.")
    return records


def split_instruction_records(
    records: list[InstructionRecord] | list[ChatRecord],
    *,
    validation_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[
    list[InstructionRecord] | list[ChatRecord],
    list[InstructionRecord] | list[ChatRecord],
]:
    """Split records by stable SHA-256 assignment while preserving order."""

    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1).")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    training: list[InstructionRecord] | list[ChatRecord] = []
    validation: list[InstructionRecord] | list[ChatRecord] = []
    for record in records:
        if isinstance(record, ChatRecord):
            record_identity = format_chat_messages(record.messages)
        else:
            record_identity = f"{record.instruction}\0{record.response}"
        identity = f"{seed}\0{record_identity}"
        digest = hashlib.sha256(identity.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        (validation if value < validation_ratio else training).append(record)
    if validation_ratio > 0 and len(records) >= 2 and not validation:
        validation.append(training.pop())
    if not training:
        raise ValueError("Instruction training split must not be empty.")
    return training, validation


class InstructionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Response-masked causal-LM examples for supervised fine-tuning."""

    def __init__(
        self,
        records: Sequence[InstructionRecord | ChatRecord],
        *,
        tokenizer: Tokenizer,
        context_length: int,
    ) -> None:
        if context_length <= 2:
            raise ValueError("context_length must be greater than 2.")
        validate_chat_tokenizer(tokenizer)
        self.removed_overlength = 0
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for value in records:
            record = (
                value.to_chat_record()
                if isinstance(value, InstructionRecord)
                else value
            )
            input_ids, labels = _encode_chat_record(
                record.messages,
                tokenizer=tokenizer,
            )
            if len(input_ids) > context_length + 1:
                self.removed_overlength += 1
                continue
            if not any(label != IGNORE_INDEX for label in labels):
                raise ValueError("Chat example has no assistant target tokens.")
            if (
                len(input_ids) > context_length + 1
                or len(input_ids) != len(labels)
            ):
                raise AssertionError("SFT packing produced invalid lengths.")
            self.examples.append(
                (
                    torch.tensor(input_ids[:-1], dtype=torch.long),
                    torch.tensor(labels[1:], dtype=torch.long),
                )
            )
        if not self.examples:
            raise ValueError("Instruction dataset must not be empty.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


def instruction_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length SFT examples while masking padded labels."""

    if not batch:
        raise ValueError("Instruction batch must not be empty.")
    if pad_token_id != 0:
        raise ValueError("Codexa instruction padding requires <pad>=0.")
    input_ids, labels = zip(*batch, strict=True)
    return (
        pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
        pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
    )


def validate_pad_token(tokenizer: Tokenizer) -> None:
    if tokenizer.token_to_id(PAD_TOKEN) != 0:
        raise ValueError("Tokenizer must use <pad>=0.")


def _encode_chat_record(
    messages: Sequence[ChatMessage],
    *,
    tokenizer: Tokenizer,
) -> tuple[list[int], list[int]]:
    """Encode one conversation and mask every non-assistant token."""

    return encode_chat_messages(
        messages,
        tokenizer=tokenizer,
        add_generation_prompt=False,
        assistant_only_labels=True,
        ignore_index=IGNORE_INDEX,
    )
