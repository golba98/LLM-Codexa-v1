"""Tests for deterministic canonical chat-dataset preparation."""

import json
from pathlib import Path
import tempfile

from src.chat_data import prepare_chat_dataset
from src.sft import load_chat_records


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source = root / "instructions.jsonl"
        rows = [
            {
                "instruction": "Name a primary color.",
                "context": "",
                "response": "Blue is a primary color.",
                "category": "closed_qa",
            },
            {
                "instruction": "Rewrite this politely.",
                "context": "Give me that.",
                "response": "Could you please give me that?",
                "category": "rewrite",
            },
        ]
        source.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        output = root / "chat.jsonl"
        manifest_path = root / "manifest.json"
        statistics, first_manifest = prepare_chat_dataset(
            [source],
            output_path=output,
            manifest_path=manifest_path,
            dataset_name="test-chat",
            license_name="test-only",
            seed=42,
            multi_turn_ratio=1.0,
        )
        assert statistics.output_records == 2
        assert statistics.multi_turn_records == 2
        assert statistics.single_turn_records == 0
        assert statistics.message_count == 8
        records = load_chat_records(output)
        assert len(records) == 2
        assert all(len(record.messages) == 4 for record in records)
        assert first_manifest["chat_template_version"] == "3.0"

        copied_output = root / "chat-copy.jsonl"
        copied_manifest = root / "manifest-copy.json"
        _, second_manifest = prepare_chat_dataset(
            [source],
            output_path=copied_output,
            manifest_path=copied_manifest,
            dataset_name="test-chat",
            license_name="test-only",
            seed=42,
            multi_turn_ratio=1.0,
        )
        assert output.read_bytes() == copied_output.read_bytes()
        assert (
            first_manifest["output_sha256"]
            == second_manifest["output_sha256"]
        )
        try:
            prepare_chat_dataset(
                [source],
                output_path=output,
                manifest_path=manifest_path,
                dataset_name="test-chat",
                license_name="test-only",
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing chat output was overwritten.")

    print("All chat-dataset tests passed.")


if __name__ == "__main__":
    main()
