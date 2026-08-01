"""Download the pinned Databricks Dolly 15k instruction dataset."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.token_data import file_sha256


REPOSITORY_ID = "databricks/databricks-dolly-15k"
REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
DATA_FILENAME = "databricks-dolly-15k.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/databricks-dolly-15k"),
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, object] = {}
    for filename in (DATA_FILENAME, "README.md"):
        path = Path(
            hf_hub_download(
                repo_id=REPOSITORY_ID,
                repo_type="dataset",
                filename=filename,
                revision=REVISION,
                local_dir=arguments.output_dir,
            )
        )
        downloaded[filename] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        print(f"Downloaded {filename}: {path.stat().st_size:,} bytes")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY_ID,
        "revision": REVISION,
        "license": "cc-by-sa-3.0",
        "source_url": (
            "https://huggingface.co/datasets/"
            "databricks/databricks-dolly-15k"
        ),
        "files": downloaded,
    }
    manifest_path = arguments.output_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
