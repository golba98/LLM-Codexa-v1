"""Export inference-only Codexa artifacts with SHA-256 checksums."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from safetensors.torch import save_model


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate import _checkpoint_model_config
from src.checkpointing import (
    load_model_checkpoint,
    verify_checkpoint_checksum,
)
from src.model import LanguageModel, count_parameters
from src.token_data import file_sha256


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def run(arguments: argparse.Namespace) -> Path:
    if arguments.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite release directory: {arguments.output_dir}"
        )
    verify_checkpoint_checksum(arguments.checkpoint)
    model_config = _checkpoint_model_config(arguments.checkpoint)
    model = LanguageModel(model_config)
    checkpoint = load_model_checkpoint(
        arguments.checkpoint,
        model=model,
        map_location="cpu",
    )
    tokenizer_checksum = file_sha256(arguments.tokenizer)
    if (
        checkpoint.tokenizer_sha256 is not None
        and checkpoint.tokenizer_sha256 != tokenizer_checksum
    ):
        raise ValueError("Tokenizer checksum does not match the checkpoint.")

    arguments.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(
            dir=arguments.output_dir.parent,
            prefix=f".{arguments.output_dir.name}.",
        )
    )
    try:
        weights_path = temporary_directory / "model.safetensors"
        save_model(
            model,
            str(weights_path),
            metadata={
                "format": "pt",
                "architecture": "CodexaLanguageModel",
                "checkpoint_optimizer_step": str(
                    checkpoint.training_state.optimizer_step
                ),
            },
        )
        shutil.copy2(arguments.tokenizer, temporary_directory / "tokenizer.json")
        _write_json(
            temporary_directory / "model_config.json",
            asdict(model_config),
        )
        _write_json(
            temporary_directory / "training_state.json",
            checkpoint.training_state.to_dict(),
        )
        for filename in ("MODEL_CARD.md", "LICENSE"):
            source = Path(filename)
            if source.is_file():
                shutil.copy2(source, temporary_directory / filename)
        if checkpoint.training_stage == "supervised_fine_tuning":
            chat_template = Path("documentation/CHAT_TEMPLATE.md")
            if not chat_template.is_file():
                raise FileNotFoundError(
                    "SFT release requires documentation/CHAT_TEMPLATE.md."
                )
            shutil.copy2(
                chat_template,
                temporary_directory / "CHAT_TEMPLATE.md",
            )

        artifact_names = [
            path.name
            for path in temporary_directory.iterdir()
            if path.is_file()
        ]
        checksums = {
            name: file_sha256(temporary_directory / name)
            for name in sorted(artifact_names)
        }
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "format_version": "1.0",
            "architecture": "CodexaLanguageModel",
            "parameter_count": count_parameters(model),
            "source_checkpoint_sha256": verify_checkpoint_checksum(
                arguments.checkpoint
            ),
            "source_checkpoint_optimizer_step": (
                checkpoint.training_state.optimizer_step
            ),
            "source_checkpoint_run_id": checkpoint.run_id,
            "training_stage": checkpoint.training_stage,
            "chat_template_version": checkpoint.chat_template_version,
            "base_checkpoint": checkpoint.base_checkpoint,
            "tokenizer_sha256": tokenizer_checksum,
            "artifacts": checksums,
        }
        _write_json(temporary_directory / "release_manifest.json", manifest)
        manifest_checksum = file_sha256(
            temporary_directory / "release_manifest.json"
        )
        (temporary_directory / "SHA256SUMS").write_text(
            "".join(
                f"{checksum}  {name}\n"
                for name, checksum in sorted(
                    {
                        **checksums,
                        "release_manifest.json": manifest_checksum,
                    }.items()
                )
            ),
            encoding="utf-8",
        )
        os.replace(temporary_directory, arguments.output_dir)
    except BaseException:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise

    print(f"Release directory: {arguments.output_dir}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Checkpoint step: {checkpoint.training_state.optimizer_step}")
    return arguments.output_dir


def main() -> None:
    arguments = build_argument_parser().parse_args()
    run(arguments)


if __name__ == "__main__":
    main()
