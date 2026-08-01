"""Tests for verified artifact backups."""

import json
from pathlib import Path
import tempfile

from scripts.backup_artifacts import backup_artifacts


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source = root / "release"
        source.mkdir()
        source.joinpath("model.bin").write_bytes(b"model bytes")
        source.joinpath("config.json").write_text(
            '{"size": 1}\n',
            encoding="utf-8",
        )
        destination = root / "backup"
        try:
            backup_artifacts([source], destination=destination)
        except ValueError:
            pass
        else:
            raise AssertionError("Same-filesystem backup should be rejected.")

        backup_artifacts(
            [source],
            destination=destination,
            allow_same_filesystem=True,
        )
        assert destination.joinpath("release/model.bin").read_bytes() == (
            b"model bytes"
        )
        manifest = json.loads(
            destination.joinpath("backup_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert "release/model.bin" in manifest["checksums"]
        try:
            backup_artifacts(
                [source],
                destination=destination,
                allow_same_filesystem=True,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing backup should be rejected.")

    print("All artifact-backup tests passed.")


if __name__ == "__main__":
    main()
