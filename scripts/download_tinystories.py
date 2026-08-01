"""Download the pinned TinyStories train/validation text files."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


REPOSITORY_ID = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
FILENAMES = (
    "TinyStories-train.txt",
    "TinyStories-valid.txt",
    "README.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/tinystories"),
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, object] = {}
    for filename in FILENAMES:
        path = Path(
            hf_hub_download(
                repo_id=REPOSITORY_ID,
                repo_type="dataset",
                filename=filename,
                revision=REVISION,
                local_dir=arguments.output_dir,
            )
        )
        files[filename] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        print(f"Downloaded {filename}: {path.stat().st_size:,} bytes")

    manifest = {
        "repository": REPOSITORY_ID,
        "revision": REVISION,
        "license": "cdla-sharing-1.0",
        "source_url": f"https://huggingface.co/datasets/{REPOSITORY_ID}",
        "files": files,
    }
    manifest_path = arguments.output_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
