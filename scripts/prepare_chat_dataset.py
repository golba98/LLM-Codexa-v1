"""Prepare a deterministic canonical JSONL dataset for Codexa chat tuning."""

import argparse
from pathlib import Path

from src.chat_data import prepare_chat_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-turn-ratio", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    statistics, manifest = prepare_chat_dataset(
        arguments.inputs,
        output_path=arguments.output,
        manifest_path=arguments.manifest,
        dataset_name=arguments.dataset_name,
        license_name=arguments.license,
        seed=arguments.seed,
        multi_turn_ratio=arguments.multi_turn_ratio,
        overwrite=arguments.overwrite,
    )
    print(f"Output records: {statistics.output_records:,}")
    print(f"Single-turn records: {statistics.single_turn_records:,}")
    print(f"Multi-turn records: {statistics.multi_turn_records:,}")
    print(f"Messages: {statistics.message_count:,}")
    print(f"SHA-256: {manifest['output_sha256']}")
    print(f"Output: {arguments.output}")
    print(f"Manifest: {arguments.manifest}")


if __name__ == "__main__":
    main()
