"""Build a clean mixed SFT dataset from instruction and conversation records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sft import ChatRecord, load_chat_records_with_statistics


PLACEHOLDER_MARKERS = (
    "[your name]",
    "[number]",
    "[insert",
    "{your name}",
    "{name}",
    "<your name>",
)
CONTAMINATION_MARKERS = (
    "cookie policy",
    "add to cart",
    "buy now",
    "sponsored content",
    "sign up for our newsletter",
    "breadcrumb navigation",
    "product sku",
    "advertisement",
)
ROLE_HEADER_PATTERN = re.compile(r"(?mi)^(system|user|assistant)\s*:")
REPEATED_PUNCTUATION_PATTERN = re.compile(r"[!?.,;:_-]{8,}")
HTML_PATTERN = re.compile(r"</?(?:html|body|nav|script|style|iframe)\b", re.I)


def contains_placeholder(record: ChatRecord) -> bool:
    """Return whether any message contains a known template placeholder."""

    return any(
        marker in message.content.lower()
        for message in record.messages
        for marker in PLACEHOLDER_MARKERS
    )


def contamination_reason(record: ChatRecord) -> str | None:
    """Return one auditable rejection reason for obvious scraped artifacts."""

    contents = "\n".join(message.content for message in record.messages)
    lowered = contents.lower()
    if any(marker in lowered for marker in CONTAMINATION_MARKERS):
        return "advertising_or_product_metadata"
    if HTML_PATTERN.search(contents):
        return "html_or_navigation"
    if REPEATED_PUNCTUATION_PATTERN.search(contents):
        return "repeated_punctuation"
    if ROLE_HEADER_PATTERN.search(contents):
        return "embedded_role_headers"
    if contains_placeholder(record):
        return "template_placeholder"
    return None


def record_key(record: ChatRecord) -> str:
    """Return a stable identity for one conversation."""

    return json.dumps(
        [(message.role, message.content) for message in record.messages],
        ensure_ascii=False,
        sort_keys=True,
    )


def write_dataset(
    input_paths: list[Path],
    output_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Filter, merge, deduplicate, and write canonical chat records."""

    records, load_statistics = load_chat_records_with_statistics(input_paths)
    output: list[ChatRecord] = []
    seen: set[str] = set()
    duplicates_removed = 0
    contamination_counts: dict[str, int] = {}
    for record in records:
        reason = contamination_reason(record)
        if reason is not None:
            contamination_counts[reason] = contamination_counts.get(reason, 0) + 1
            continue
        key = record_key(record)
        if key in seen:
            duplicates_removed += 1
            continue
        seen.add(key)
        output.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for ordinal, record in enumerate(output):
            value = {
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in record.messages
                ],
                "category": record.category,
                "source": record.source,
                "conversation_id": record.conversation_id or f"clean-{ordinal:08d}",
            }
            output_file.write(json.dumps(value, ensure_ascii=False) + "\n")
    statistics = {
        "input_records": len(records),
        "output_records": len(output),
        "duplicates_removed": duplicates_removed,
        "loader": load_statistics.to_dict(),
        "contamination_removed": sum(contamination_counts.values()),
        "contamination_reasons": dict(sorted(contamination_counts.items())),
        "removed_samples": (
            load_statistics.removed_samples
            + duplicates_removed
            + sum(contamination_counts.values())
        ),
        "multi_turn_records": sum(len(record.messages) >= 4 for record in output),
    }
    manifest_path.write_text(json.dumps(statistics, indent=2) + "\n", encoding="utf-8")
    return statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(write_dataset(arguments.input, arguments.output, arguments.manifest), indent=2))


if __name__ == "__main__":
    main()
