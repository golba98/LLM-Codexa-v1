"""Instruction formatting and supervised fine-tuning datasets."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from src.tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


CHAT_TEMPLATE_VERSION = "1.0"
IGNORE_INDEX = -100


@dataclass(frozen=True)
class InstructionRecord:
    instruction: str
    context: str
    response: str
    category: str


def format_instruction_prompt(instruction: str, context: str = "") -> str:
    """Format a user instruction while leaving the response empty."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string.")
    if not isinstance(context, str):
        raise ValueError("context must be a string.")
    sections = [f"User: {instruction.strip()}"]
    if context.strip():
        sections.append(f"Context: {context.strip()}")
    sections.append("Assistant:")
    return "\n".join(sections)


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
    records: list[InstructionRecord],
    *,
    validation_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[InstructionRecord], list[InstructionRecord]]:
    """Split records by stable SHA-256 assignment while preserving order."""

    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1).")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    training: list[InstructionRecord] = []
    validation: list[InstructionRecord] = []
    for ordinal, record in enumerate(records):
        identity = (
            f"{seed}\0{ordinal}\0{record.instruction}\0{record.response}"
        )
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
        records: list[InstructionRecord],
        *,
        tokenizer: Tokenizer,
        context_length: int,
    ) -> None:
        if context_length <= 2:
            raise ValueError("context_length must be greater than 2.")
        bos_id = tokenizer.token_to_id(BOS_TOKEN)
        eos_id = tokenizer.token_to_id(EOS_TOKEN)
        if bos_id != 1 or eos_id != 2:
            raise ValueError("Tokenizer must use <bos>=1 and <eos>=2.")
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for record in records:
            prompt_ids = tokenizer.encode(
                format_instruction_prompt(
                    record.instruction,
                    record.context,
                ),
                add_special_tokens=False,
            ).ids
            response_ids = tokenizer.encode(
                " " + record.response,
                add_special_tokens=False,
            ).ids
            maximum_prompt = max(1, context_length - 2)
            if len(prompt_ids) > maximum_prompt:
                prefix_length = maximum_prompt // 2
                prompt_ids = (
                    prompt_ids[:prefix_length]
                    + prompt_ids[-(maximum_prompt - prefix_length) :]
                )
            available_response = context_length - 1 - len(prompt_ids)
            response_ids = response_ids[: max(0, available_response - 1)]
            target_ids = [*response_ids, eos_id]
            input_ids = [bos_id, *prompt_ids, *target_ids]
            labels = [
                *([IGNORE_INDEX] * (1 + len(prompt_ids))),
                *target_ids,
            ]
            if len(input_ids) > context_length or len(input_ids) != len(labels):
                raise AssertionError("SFT packing produced invalid lengths.")
            self.examples.append(
                (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long),
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
