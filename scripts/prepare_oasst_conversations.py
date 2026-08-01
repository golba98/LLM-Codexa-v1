"""Prepare quality-filtered multi-turn OASST1 conversation chains."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset


def _quality(row: dict[str, Any]) -> float:
    labels = row.get("labels") or {}
    names = labels.get("name", [])
    values = labels.get("value", [])
    scores = dict(zip(names, values))
    return float(scores.get("quality", 0.0))


def _usable(row: dict[str, Any], *, minimum_quality: float) -> bool:
    return (
        row.get("lang") == "en"
        and row.get("deleted") is False
        and row.get("tree_state") == "ready_for_export"
        and row.get("review_result") is True
        and _quality(row) >= minimum_quality
        and isinstance(row.get("text"), str)
        and bool(row["text"].strip())
    )


def prepare(
    input_paths: list[Path],
    output_path: Path,
    manifest_path: Path,
    *,
    minimum_quality: float,
    maximum_records: int,
) -> dict[str, int]:
    rows: list[dict[str, Any]] = []
    for input_path in input_paths:
        dataset = load_dataset("parquet", data_files=str(input_path), split="train")
        rows.extend(dict(row) for row in dataset)

    by_id = {
        row["message_id"]: row
        for row in rows
        if isinstance(row.get("message_id"), str)
        and _usable(row, minimum_quality=minimum_quality)
    }
    children: dict[str, list[str]] = defaultdict(list)
    for row in by_id.values():
        parent_id = row.get("parent_id")
        if parent_id in by_id:
            children[parent_id].append(row["message_id"])

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for message_id, row in by_id.items():
        if children.get(message_id):
            continue
        chain: list[dict[str, str]] = []
        current: str | None = message_id
        while current is not None and current in by_id:
            current_row = by_id[current]
            role = "user" if current_row["role"] == "prompter" else "assistant"
            text = current_row["text"].strip()
            if len(text) > 6000:
                text = text[:6000].rsplit(" ", 1)[0] + "..."
            chain.append({"role": role, "content": text})
            current = current_row.get("parent_id")
        chain.reverse()
        if (
            len(chain) < 4
            or chain[0]["role"] != "user"
            or chain[-1]["role"] != "assistant"
        ):
            continue
        if any(chain[index]["role"] == chain[index - 1]["role"] for index in range(1, len(chain))):
            continue
        identity = json.dumps(chain, ensure_ascii=False, sort_keys=True)
        if identity in seen:
            continue
        seen.add(identity)
        records.append(
            {
                "messages": chain,
                "category": "oasst1:multi_turn",
                "source": "OpenAssistant/oasst1",
                "conversation_id": f"oasst1-{row['message_tree_id']}-{message_id}",
            }
        )
        if len(records) >= maximum_records:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": "OpenAssistant/oasst1",
        "license": "Apache-2.0",
        "minimum_quality": minimum_quality,
        "input_rows": len(rows),
        "usable_rows": len(by_id),
        "multi_turn_records": len(records),
        "minimum_messages": 4,
        "output": str(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"input_rows": len(rows), "usable_rows": len(by_id), "multi_turn_records": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--minimum-quality", type=float, default=0.7)
    parser.add_argument("--maximum-records", type=int, default=10000)
    arguments = parser.parse_args()
    statistics = prepare(
        arguments.input,
        arguments.output,
        arguments.manifest,
        minimum_quality=arguments.minimum_quality,
        maximum_records=arguments.maximum_records,
    )
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
