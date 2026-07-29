"""Stream, clean, deduplicate, and convert TinyStories to JSONL."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from src.data.cleaning import CLEANING_VERSION, clean_text, text_sha256


END_MARKER = "<|endoftext|>"


def _iter_stories(path: Path):
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip() == END_MARKER:
                yield "".join(lines)
                lines.clear()
            else:
                lines.append(line)
    if lines:
        yield "".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=final_path.parent,
        prefix=f".{final_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(temporary_name)


def _write_split(
    input_path: Path,
    output_path: Path,
    *,
    split_name: str,
    seen_digests: set[str],
) -> dict[str, int | float]:
    raw_count = 0
    clean_count = 0
    empty_count = 0
    duplicate_count = 0
    characters = 0
    utf8_bytes = 0
    minimum_length: int | None = None
    maximum_length = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for raw_count, story in enumerate(_iter_stories(input_path), start=1):
            cleaned = clean_text(story)
            if cleaned is None:
                empty_count += 1
                continue
            digest = text_sha256(cleaned)
            if digest in seen_digests:
                duplicate_count += 1
                continue
            seen_digests.add(digest)
            ordinal = clean_count
            record = {
                "text": cleaned,
                "source": "roneneldan/TinyStories",
                "document_id": f"{split_name}-{ordinal:09d}",
                "metadata": {
                    "upstream_split": split_name,
                    "upstream_revision": (
                        "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
                    ),
                },
            }
            output_file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            length = len(cleaned)
            clean_count += 1
            characters += length
            utf8_bytes += len(cleaned.encode("utf-8"))
            minimum_length = (
                length if minimum_length is None else min(minimum_length, length)
            )
            maximum_length = max(maximum_length, length)
    return {
        "raw_document_count": raw_count,
        "cleaned_document_count": clean_count,
        "empty_documents_removed": empty_count,
        "duplicate_documents_removed": duplicate_count,
        "characters": characters,
        "utf8_bytes": utf8_bytes,
        "minimum_document_length": minimum_length or 0,
        "maximum_document_length": maximum_length,
        "mean_document_length": characters / clean_count if clean_count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/tinystories"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/tinystories"),
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    train_input = arguments.raw_dir / "TinyStories-train.txt"
    validation_input = arguments.raw_dir / "TinyStories-valid.txt"
    train_output = arguments.output_dir / "train.jsonl"
    validation_output = arguments.output_dir / "validation.jsonl"
    manifest_output = arguments.output_dir / "dataset_manifest.json"
    statistics_output = arguments.output_dir / "dataset_stats.json"
    final_paths = (
        train_output,
        validation_output,
        manifest_output,
        statistics_output,
    )
    existing = [path for path in final_paths if path.exists()]
    if existing and not arguments.overwrite:
        raise FileExistsError(
            "Refusing to overwrite TinyStories outputs: "
            + ", ".join(str(path) for path in existing)
        )

    temporary = {path: _temporary_path(path) for path in final_paths}
    try:
        seen_digests: set[str] = set()
        train_stats = _write_split(
            train_input,
            temporary[train_output],
            split_name="train",
            seen_digests=seen_digests,
        )
        validation_stats = _write_split(
            validation_input,
            temporary[validation_output],
            split_name="validation",
            seen_digests=seen_digests,
        )
        statistics = {
            "train": train_stats,
            "validation": validation_stats,
            "total_cleaned_documents": (
                train_stats["cleaned_document_count"]
                + validation_stats["cleaned_document_count"]
            ),
            "total_characters": (
                train_stats["characters"] + validation_stats["characters"]
            ),
            "total_utf8_bytes": (
                train_stats["utf8_bytes"] + validation_stats["utf8_bytes"]
            ),
        }
        temporary[statistics_output].write_text(
            json.dumps(statistics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_name": "roneneldan/TinyStories",
            "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
            "license": "cdla-sharing-1.0",
            "cleaning_version": CLEANING_VERSION,
            "input_paths": {
                "train": str(train_input),
                "validation": str(validation_input),
            },
            "input_checksums": {
                "train": _sha256(train_input),
                "validation": _sha256(validation_input),
            },
            "output_paths": {
                "train": str(train_output),
                "validation": str(validation_output),
                "statistics": str(statistics_output),
            },
            "output_checksums": {
                "train": _sha256(temporary[train_output]),
                "validation": _sha256(temporary[validation_output]),
            },
            "statistics": statistics,
        }
        temporary[manifest_output].write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for final_path in final_paths:
            temporary[final_path].replace(final_path)
    except BaseException:
        for temporary_path in temporary.values():
            temporary_path.unlink(missing_ok=True)
        raise
    print(json.dumps(statistics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
