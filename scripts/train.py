"""Train Codexa from a prepared memory-mapped token dataset."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import torch


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.model import LanguageModel, count_parameters
from src.token_data import (
    MemmapTokenDataset,
    create_token_dataloader,
    file_sha256,
)
from src.training import (
    JsonlRunLogger,
    create_adamw_optimizer,
    create_grad_scaler,
    resolve_device,
    resolve_precision,
    set_deterministic_seed,
    train_model,
    TrainingState,
    utc_timestamp,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic mixed-precision language-model training."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-token-file", type=Path, required=True)
    parser.add_argument("--validation-token-file", type=Path, required=True)
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--precision",
        choices=("config", "fp32", "fp16", "bf16"),
        default="config",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-validation-batches", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--run-name", default="training")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--overwrite-log", action="store_true")
    return parser


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed token manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("Token manifest must contain a JSON object.")
    return value


def _required_mapping(
    manifest: dict[str, object],
    key: str,
) -> dict[str, object]:
    value = manifest.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Token manifest field {key!r} must be an object.")
    return value


def _validate_token_file(
    path: Path,
    *,
    dtype: np.dtype,
    expected_checksum: object,
    model_vocab_size: int,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Token file does not exist: {path}")
    if path.stat().st_size % dtype.itemsize != 0:
        raise ValueError(f"Token file ends with a partial {dtype.name} element.")
    if not isinstance(expected_checksum, str):
        raise ValueError("Token manifest checksum must be a string.")
    actual_checksum = file_sha256(path)
    if actual_checksum != expected_checksum:
        raise ValueError(
            f"Token file checksum mismatch for {path}: "
            f"expected {expected_checksum}, got {actual_checksum}."
        )
    tokens = np.memmap(path, mode="r", dtype=dtype)
    if len(tokens) > 0 and int(tokens.max()) >= model_vocab_size:
        raise ValueError(
            f"Token file {path} contains an ID outside model vocabulary "
            f"size {model_vocab_size}."
        )


def validate_manifest(
    manifest: dict[str, object],
    *,
    train_token_file: Path,
    validation_token_file: Path | None,
    context_length: int,
    model_vocab_size: int,
) -> np.dtype:
    try:
        dtype = np.dtype(manifest["dtype"])
    except (KeyError, TypeError) as error:
        raise ValueError("Token manifest has an invalid dtype.") from error
    if dtype not in {np.dtype(np.uint16), np.dtype(np.uint32)}:
        raise ValueError("Token manifest dtype must be uint16 or uint32.")
    if manifest.get("context_length") != context_length:
        raise ValueError(
            "Token-data context length does not match the model configuration."
        )
    tokenizer_vocab_size = manifest.get("tokenizer_actual_vocab_size")
    if (
        not isinstance(tokenizer_vocab_size, int)
        or tokenizer_vocab_size <= 0
        or tokenizer_vocab_size > model_vocab_size
    ):
        raise ValueError(
            "Tokenizer vocabulary in the manifest is invalid for the model."
        )
    if manifest.get("model_vocab_size") != model_vocab_size:
        raise ValueError(
            "Token-data model vocabulary does not match configuration."
        )
    checksums = _required_mapping(manifest, "output_checksums")
    _validate_token_file(
        train_token_file,
        dtype=dtype,
        expected_checksum=checksums.get("train"),
        model_vocab_size=model_vocab_size,
    )
    if validation_token_file is not None:
        _validate_token_file(
            validation_token_file,
            dtype=dtype,
            expected_checksum=checksums.get("validation"),
            model_vocab_size=model_vocab_size,
        )
    return dtype


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def run(arguments: argparse.Namespace) -> int:
    project_config = load_config(arguments.config)
    training_config = project_config.training
    if arguments.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if arguments.max_validation_batches <= 0:
        raise ValueError("--max-validation-batches must be positive.")
    max_steps = (
        training_config.max_steps
        if arguments.max_steps is None
        else arguments.max_steps
    )
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

    device = resolve_device(arguments.device)
    requested_precision = (
        training_config.precision
        if arguments.precision == "config"
        else arguments.precision
    )
    precision = resolve_precision(requested_precision, device)
    set_deterministic_seed(training_config.seed)

    manifest = _load_manifest(arguments.token_manifest)
    validation_path = (
        None if arguments.no_validation else arguments.validation_token_file
    )
    dtype = validate_manifest(
        manifest,
        train_token_file=arguments.train_token_file,
        validation_token_file=validation_path,
        context_length=project_config.model.context_length,
        model_vocab_size=project_config.model.vocab_size,
    )
    train_dataset = MemmapTokenDataset(
        arguments.train_token_file,
        dtype=dtype,
        context_length=project_config.model.context_length,
        model_vocab_size=project_config.model.vocab_size,
    )
    pin_memory = device.type == "cuda"
    train_loader = create_token_dataloader(
        train_dataset,
        batch_size=training_config.micro_batch_size,
        shuffle=True,
        num_workers=arguments.num_workers,
        pin_memory=pin_memory,
        seed=training_config.seed,
    )

    validation_loader = None
    if validation_path is not None:
        validation_dataset = MemmapTokenDataset(
            validation_path,
            dtype=dtype,
            context_length=project_config.model.context_length,
            model_vocab_size=project_config.model.vocab_size,
        )
        validation_loader = create_token_dataloader(
            validation_dataset,
            batch_size=training_config.micro_batch_size,
            shuffle=False,
            num_workers=arguments.num_workers,
            pin_memory=pin_memory,
            seed=training_config.seed,
        )

    model = LanguageModel(project_config.model)
    total_parameters = count_parameters(model)
    trainable_parameters = count_parameters(model, trainable_only=True)
    model.to(device)
    optimizer = create_adamw_optimizer(
        model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scaler = create_grad_scaler(device, precision)
    run_id = str(uuid.uuid4())
    start_timestamp = utc_timestamp()

    print(f"Device: {device}")
    print(f"Precision: {precision.name}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    logger = JsonlRunLogger(
        arguments.log_dir,
        arguments.run_name,
        overwrite=arguments.overwrite_log,
    )
    state = TrainingState()
    try:
        logger.write_metadata(
            {
                "run_name": arguments.run_name,
                "run_id": run_id,
                "config_path": str(arguments.config),
                "train_token_file_path": str(arguments.train_token_file),
                "validation_token_file_path": (
                    None if validation_path is None else str(validation_path)
                ),
                "token_manifest_path": str(arguments.token_manifest),
                "device": str(device),
                "precision": precision.name,
                "start_timestamp_utc": start_timestamp,
                "requested_maximum_optimizer_steps": max_steps,
                "gradient_accumulation_steps": (
                    training_config.gradient_accumulation_steps
                ),
                "micro_batch_size": training_config.micro_batch_size,
                "context_length": project_config.model.context_length,
                "total_parameter_count": total_parameters,
                "trainable_parameter_count": trainable_parameters,
                "git_commit": _git_commit(),
                "command_line_arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(arguments).items()
                },
                "model_config": asdict(project_config.model),
                "training_config": asdict(training_config),
            }
        )
        state, metrics = train_model(
            model,
            train_loader,
            optimizer,
            device=device,
            precision=precision,
            max_steps=max_steps,
            gradient_accumulation_steps=(
                training_config.gradient_accumulation_steps
            ),
            gradient_clip=training_config.gradient_clip,
            warmup_steps=min(training_config.warmup_steps, max_steps - 1),
            peak_learning_rate=training_config.learning_rate,
            seed=training_config.seed,
            state=state,
            scaler=scaler,
            validation_loader=validation_loader,
            evaluation_interval=(
                None
                if validation_loader is None
                else training_config.evaluation_interval
            ),
            max_validation_batches=arguments.max_validation_batches,
            run_name=arguments.run_name,
            run_id=run_id,
            logger=logger,
            progress=True,
        )
    except KeyboardInterrupt:
        logger.flush()
        print(
            f"Interrupted after {state.optimizer_step} completed optimizer steps.",
            file=sys.stderr,
        )
        return 130
    finally:
        logger.close()

    final = metrics[-1]
    print(f"Completed optimizer steps: {state.optimizer_step}")
    print(f"Completed micro-steps: {state.micro_step}")
    print(f"Tokens processed: {state.tokens_seen}")
    print(f"Final loss: {final.training_loss:.6f}")
    print(f"Metrics: {logger.metrics_path}")
    print(f"Run metadata: {logger.metadata_path}")
    return 0


def main() -> None:
    arguments = build_argument_parser().parse_args()
    raise SystemExit(run(arguments))


if __name__ == "__main__":
    main()
