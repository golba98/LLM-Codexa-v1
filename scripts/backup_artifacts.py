"""Create a verified, atomic backup of model release artifacts."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.token_data import file_sha256


def _existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise FileNotFoundError(f"No existing ancestor for {path}.")
        current = current.parent
    return current


def backup_artifacts(
    sources: list[Path],
    *,
    destination: Path,
    allow_same_filesystem: bool = False,
) -> Path:
    """Copy files/directories and verify every copied byte by SHA-256."""

    if not sources:
        raise ValueError("At least one backup source is required.")
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite backup destination: {destination}"
        )
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"Backup source does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination_device = _existing_ancestor(destination.parent).stat().st_dev
    if not allow_same_filesystem:
        same_device = [
            source
            for source in sources
            if source.stat().st_dev == destination_device
        ]
        if same_device:
            raise ValueError(
                "Backup destination must use a different filesystem from all "
                "sources; use --allow-same-filesystem only for testing."
            )

    temporary = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
        )
    )
    try:
        copied_roots: list[str] = []
        for source in sources:
            target = temporary / source.name
            if target.exists():
                raise ValueError(
                    f"Backup sources have a duplicate basename: {source.name}"
                )
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
            copied_roots.append(source.name)

        checksums: dict[str, str] = {}
        for path in sorted(temporary.rglob("*")):
            if path.is_file():
                relative = str(path.relative_to(temporary))
                checksums[relative] = file_sha256(path)
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "format_version": "1.0",
            "source_paths": [str(source) for source in sources],
            "copied_roots": copied_roots,
            "checksums": checksums,
            "independent_filesystem_required": not allow_same_filesystem,
        }
        (temporary / "backup_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for relative, expected in checksums.items():
            actual = file_sha256(temporary / relative)
            if actual != expected:
                raise IOError(f"Backup verification failed for {relative}.")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--allow-same-filesystem",
        action="store_true",
        help="Permit a non-independent copy for tests only.",
    )
    arguments = parser.parse_args()
    result = backup_artifacts(
        arguments.sources,
        destination=arguments.destination,
        allow_same_filesystem=arguments.allow_same_filesystem,
    )
    print(f"Verified backup: {result}")


if __name__ == "__main__":
    main()
