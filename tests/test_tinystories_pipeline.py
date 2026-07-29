"""Tests for the pinned TinyStories conversion helpers."""

import json
from pathlib import Path
import tempfile

from scripts.prepare_tinystories import END_MARKER, _iter_stories, _write_split


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        raw_path = root / "TinyStories-train.txt"
        output_path = root / "train.jsonl"
        raw_path.write_text(
            " First  story.  \n"
            f"{END_MARKER}\n"
            "\x00\n"
            f"{END_MARKER}\n"
            " First story.\n"
            f"{END_MARKER}\n"
            "Second\tstory.\n",
            encoding="utf-8",
        )

        stories = list(_iter_stories(raw_path))
        assert len(stories) == 4
        statistics = _write_split(
            raw_path,
            output_path,
            split_name="train",
            seen_digests=set(),
        )
        records = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
        ]

        assert statistics["raw_document_count"] == 4
        assert statistics["cleaned_document_count"] == 2
        assert statistics["empty_documents_removed"] == 1
        assert statistics["duplicate_documents_removed"] == 1
        assert [record["text"] for record in records] == [
            "First story.",
            "Second story.",
        ]
        assert [record["document_id"] for record in records] == [
            "train-000000000",
            "train-000000001",
        ]
        assert all(
            record["source"] == "roneneldan/TinyStories"
            for record in records
        )

    print("All TinyStories pipeline tests passed.")


if __name__ == "__main__":
    main()
