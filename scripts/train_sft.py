"""Supervised fine-tune Codexa on response-masked instruction data."""

import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import sys
import uuid

import torch
from torch.utils.data import DataLoader


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.checkpointing import (
    CheckpointManager,
    SchedulerState,
    build_checkpoint_payload,
    load_checkpoint,
    load_model_checkpoint,
)
from src.config import load_config
from src.model import LanguageModel, count_parameters
from src.sft import (
    CHAT_TEMPLATE_VERSION,
    InstructionDataset,
    instruction_collate,
    load_chat_records,
    split_instruction_records,
    validate_pad_token,
)
from src.token_data import file_sha256
from src.tokenizer import load_tokenizer
from src.training import (
    JsonlRunLogger,
    TrainingState,
    create_adamw_optimizer,
    create_grad_scaler,
    resolve_device,
    resolve_precision,
    set_deterministic_seed,
    train_model,
    utc_timestamp,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-checkpoint", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--instruction-jsonl", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("config", "fp32", "fp16", "bf16"),
        default="config",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-validation-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--run-name", default="sft")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--overwrite-log", action="store_true")
    return parser


def run(arguments: argparse.Namespace) -> int:
    if arguments.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if arguments.max_validation_batches <= 0:
        raise ValueError("--max-validation-batches must be positive.")
    project_config = load_config(arguments.config)
    training_config = project_config.training
    max_steps = (
        training_config.max_steps
        if arguments.max_steps is None
        else arguments.max_steps
    )
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    device = resolve_device(arguments.device)
    precision_name = (
        training_config.precision
        if arguments.precision == "config"
        else arguments.precision
    )
    precision = resolve_precision(precision_name, device)
    set_deterministic_seed(training_config.seed)

    tokenizer = load_tokenizer(arguments.tokenizer)
    validate_pad_token(tokenizer)
    tokenizer_checksum = file_sha256(arguments.tokenizer)
    records = load_chat_records(arguments.instruction_jsonl)
    training_records, validation_records = split_instruction_records(
        records,
        validation_ratio=arguments.validation_ratio,
        seed=training_config.seed,
    )
    if not validation_records:
        raise ValueError("SFT validation split must not be empty.")
    train_dataset = InstructionDataset(
        training_records,
        tokenizer=tokenizer,
        context_length=project_config.model.context_length,
    )
    validation_dataset = InstructionDataset(
        validation_records,
        tokenizer=tokenizer,
        context_length=project_config.model.context_length,
    )
    collate = partial(instruction_collate, pad_token_id=0)
    generator = torch.Generator()
    generator.manual_seed(training_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.micro_batch_size,
        shuffle=True,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_config.micro_batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    model = LanguageModel(project_config.model).to(device)
    optimizer = create_adamw_optimizer(
        model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scaler = create_grad_scaler(device, precision)
    scheduler = SchedulerState(
        warmup_steps=min(training_config.warmup_steps, max_steps - 1),
        max_steps=max_steps,
        peak_learning_rate=training_config.learning_rate,
        minimum_learning_rate=training_config.learning_rate * 0.1,
    )
    state = TrainingState()
    run_id = str(uuid.uuid4())
    base_checkpoint_metadata: dict[str, object] | None = None
    if arguments.resume is not None:
        if arguments.overwrite_log:
            raise ValueError("--resume cannot be combined with --overwrite-log.")
        loaded = load_checkpoint(
            arguments.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_config=project_config,
            restore_rng=True,
            map_location=device,
        )
        if loaded.scheduler != scheduler:
            raise ValueError("SFT checkpoint schedule does not match this run.")
        if loaded.run_name != arguments.run_name:
            raise ValueError("SFT checkpoint run name does not match.")
        state = loaded.state
        run_id = loaded.run_id
        metadata_path = (
            arguments.log_dir / arguments.run_name / "run_metadata.json"
        )
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            existing_base = metadata.get("base_checkpoint")
            if isinstance(existing_base, dict):
                base_checkpoint_metadata = existing_base
    else:
        assert arguments.base_checkpoint is not None
        base = load_model_checkpoint(
            arguments.base_checkpoint,
            model=model,
            map_location=device,
        )
        expected_model = asdict(project_config.model)
        if base.config.get("model") != expected_model:
            raise ValueError("Base checkpoint model geometry does not match SFT.")
        if (
            base.tokenizer_sha256 is not None
            and base.tokenizer_sha256 != tokenizer_checksum
        ):
            raise ValueError("Base checkpoint tokenizer does not match SFT.")
        base_checkpoint_metadata = {
            "path": str(arguments.base_checkpoint),
            "run_name": base.run_name,
            "run_id": base.run_id,
            "optimizer_step": base.training_state.optimizer_step,
        }

    logger = JsonlRunLogger(
        arguments.log_dir,
        arguments.run_name,
        overwrite=arguments.overwrite_log,
        resume=arguments.resume is not None,
        expected_run_id=run_id,
        expected_optimizer_step=(
            state.optimizer_step if arguments.resume is not None else None
        ),
    )
    manager = CheckpointManager(arguments.checkpoint_dir, arguments.run_name)

    def save_checkpoint(current_state, metrics, *, force: bool = False) -> None:
        step = current_state.optimizer_step
        interval = step % training_config.checkpoint_interval == 0
        validation_loss = getattr(metrics, "validation_loss", None)
        is_best = (
            validation_loss is not None
            and current_state.best_validation_loss == validation_loss
        )
        if not force and not interval and step != max_steps and not is_best:
            return
        payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            state=current_state,
            scheduler=scheduler,
            config=project_config,
            run_name=arguments.run_name,
            run_id=run_id,
            tokenizer_reference=str(arguments.tokenizer),
            tokenizer_sha256=tokenizer_checksum,
        )
        payload["training_stage"] = "supervised_fine_tuning"
        payload["chat_template_version"] = CHAT_TEMPLATE_VERSION
        payload["base_checkpoint"] = base_checkpoint_metadata
        manager.save(payload, is_best=is_best, milestone=interval)

    print(f"Device: {device}")
    print(f"Precision: {precision.name}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Training records: {len(training_records):,}")
    print(f"Validation records: {len(validation_records):,}")
    try:
        if arguments.resume is None:
            logger.write_metadata(
                {
                    "run_name": arguments.run_name,
                    "run_id": run_id,
                    "training_stage": "supervised_fine_tuning",
                    "chat_template_version": CHAT_TEMPLATE_VERSION,
                    "start_timestamp_utc": utc_timestamp(),
                    "config_path": str(arguments.config),
                    "instruction_jsonl": str(arguments.instruction_jsonl),
                    "instruction_sha256": file_sha256(
                        arguments.instruction_jsonl
                    ),
                    "instruction_record_count": len(records),
                    "training_record_count": len(training_records),
                    "validation_record_count": len(validation_records),
                    "validation_ratio": arguments.validation_ratio,
                    "tokenizer": str(arguments.tokenizer),
                    "tokenizer_sha256": tokenizer_checksum,
                    "base_checkpoint": base_checkpoint_metadata,
                    "device": str(device),
                    "precision": precision.name,
                    "requested_maximum_optimizer_steps": max_steps,
                    "model_config": asdict(project_config.model),
                    "training_config": asdict(training_config),
                }
            )
        if state.optimizer_step >= max_steps:
            print("SFT checkpoint already completed the requested run.")
            return 0
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
            warmup_steps=scheduler.warmup_steps,
            peak_learning_rate=scheduler.peak_learning_rate,
            minimum_learning_rate=scheduler.minimum_learning_rate,
            seed=training_config.seed,
            state=state,
            scaler=scaler,
            validation_loader=validation_loader,
            evaluation_interval=training_config.evaluation_interval,
            max_validation_batches=arguments.max_validation_batches,
            run_name=arguments.run_name,
            run_id=run_id,
            logger=logger,
            on_optimizer_step=save_checkpoint,
            progress=True,
        )
    except KeyboardInterrupt:
        logger.flush()
        at_boundary = (
            state.micro_step
            == state.optimizer_step
            * training_config.gradient_accumulation_steps
        )
        if state.optimizer_step > 0 and at_boundary:
            save_checkpoint(state, object(), force=True)
        print(
            f"Interrupted after {state.optimizer_step} SFT optimizer steps.",
            file=sys.stderr,
        )
        return 130
    finally:
        logger.close()
    final = metrics[-1]
    print(f"Completed optimizer steps: {state.optimizer_step}")
    print(f"Final loss: {final.training_loss:.6f}")
    print(f"Metrics: {logger.metrics_path}")
    return 0


def main() -> None:
    arguments = build_argument_parser().parse_args()
    raise SystemExit(run(arguments))


if __name__ == "__main__":
    main()
