"""Deterministic preparation of canonical Codexa chat-training data."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from src.sft import ChatMessage, ChatRecord, load_chat_records
from src.token_data import file_sha256


CHAT_DATA_FORMAT_VERSION = "1.0"


@dataclass(frozen=True)
class ChatDatasetStatistics:
    """Auditable counts for one prepared chat dataset."""

    input_records: int
    output_records: int
    duplicate_records_removed: int
    single_turn_records: int
    multi_turn_records: int
    message_count: int
    category_counts: dict[str, int]
    source_counts: dict[str, int]


def augment_with_context_turn(
    record: ChatRecord,
    *,
    ordinal: int,
    seed: int,
    ratio: float,
) -> ChatRecord:
    """Add a deterministic, answerable context-recall turn to some records."""

    if not 0 <= ratio <= 1:
        raise ValueError("multi_turn_ratio must be in [0, 1].")
    identity = (
        f"{seed}\0{ordinal}\0"
        + "\0".join(
            f"{message.role}\0{message.content}" for message in record.messages
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    selected = int.from_bytes(digest[:8], "big") / 2**64 < ratio
    if not selected or len(record.messages) > 2:
        return record
    original_user = record.messages[-2].content
    if digest[8] % 2 == 0:
        follow_up = ChatMessage(
            "user",
            "What was my original request? Answer in one sentence.",
        )
        follow_up_answer = ChatMessage(
            "assistant",
            f'Your original request was: "{original_user}"',
        )
    else:
        follow_up = ChatMessage(
            "user",
            "Repeat your previous answer exactly.",
        )
        follow_up_answer = ChatMessage(
            "assistant",
            record.messages[-1].content,
        )
    return ChatRecord(
        messages=(*record.messages, follow_up, follow_up_answer),
        category=f"{record.category}:multi_turn",
        source=record.source,
        conversation_id=record.conversation_id,
    )


def prepare_chat_dataset(
    input_paths: list[str | Path],
    *,
    output_path: str | Path,
    manifest_path: str | Path,
    dataset_name: str,
    license_name: str,
    seed: int = 42,
    multi_turn_ratio: float = 0.2,
    overwrite: bool = False,
) -> tuple[ChatDatasetStatistics, dict[str, object]]:
    """Normalize, deduplicate, augment, and atomically write chat JSONL."""

    if not input_paths:
        raise ValueError("At least one chat input path is required.")
    if not dataset_name.strip():
        raise ValueError("dataset_name must not be empty.")
    if not license_name.strip():
        raise ValueError("license_name must not be empty.")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not 0 <= multi_turn_ratio <= 1:
        raise ValueError("multi_turn_ratio must be in [0, 1].")
    destination = Path(output_path)
    manifest_destination = Path(manifest_path)
    for target in (destination, manifest_destination):
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)

    raw_records: list[ChatRecord] = []
    for input_path in input_paths:
        raw_records.extend(load_chat_records(input_path))
    output_records: list[ChatRecord] = []
    seen: set[str] = set()
    duplicate_count = 0
    for ordinal, record in enumerate(raw_records):
        prepared = augment_with_context_turn(
            record,
            ordinal=ordinal,
            seed=seed,
            ratio=multi_turn_ratio,
        )
        identity = hashlib.sha256(
            json.dumps(
                [
                    {"role": message.role, "content": message.content}
                    for message in prepared.messages
                ],
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if identity in seen:
            duplicate_count += 1
            continue
        seen.add(identity)
        output_records.append(prepared)

    _write_records_atomic(destination, output_records)
    category_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for record in output_records:
        category_counts[record.category] = (
            category_counts.get(record.category, 0) + 1
        )
        source_counts[record.source] = source_counts.get(record.source, 0) + 1
    statistics = ChatDatasetStatistics(
        input_records=len(raw_records),
        output_records=len(output_records),
        duplicate_records_removed=duplicate_count,
        single_turn_records=sum(
            len(record.messages) == 2 for record in output_records
        ),
        multi_turn_records=sum(
            len(record.messages) > 2 for record in output_records
        ),
        message_count=sum(len(record.messages) for record in output_records),
        category_counts=dict(sorted(category_counts.items())),
        source_counts=dict(sorted(source_counts.items())),
    )
    manifest: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format_version": CHAT_DATA_FORMAT_VERSION,
        "chat_template_version": "2.0",
        "dataset_name": dataset_name,
        "license": license_name,
        "seed": seed,
        "multi_turn_ratio": multi_turn_ratio,
        "input_paths": [str(Path(path)) for path in input_paths],
        "input_sha256": {
            str(Path(path)): file_sha256(path) for path in input_paths
        },
        "output_path": str(destination),
        "output_sha256": file_sha256(destination),
        "statistics": asdict(statistics),
    }
    _write_json_atomic(manifest_destination, manifest)
    return statistics, manifest


def _write_records_atomic(path: Path, records: list[ChatRecord]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as output_file:
        temporary_path = Path(output_file.name)
        try:
            for ordinal, record in enumerate(records):
                value = {
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in record.messages
                    ],
                    "category": record.category,
                    "source": record.source,
                    "conversation_id": (
                        record.conversation_id or f"chat-{ordinal:08d}"
                    ),
                }
                output_file.write(
                    json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
                )
            output_file.flush()
            os.fsync(output_file.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as output_file:
        temporary_path = Path(output_file.name)
        try:
            json.dump(value, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
