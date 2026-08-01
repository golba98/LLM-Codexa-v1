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
    verify_checkpoint_checksum,
)
from src.config import load_config
from src.chat_protocol import (
    CHAT_VOCAB_SIZE,
    chat_special_token_map,
    validate_chat_tokenizer,
)
from src.model import LanguageModel, ModelConfig, count_parameters
from src.sft import (
    CHAT_TEMPLATE_VERSION,
    InstructionDataset,
    format_chat_messages,
    instruction_collate,
    load_chat_records_with_statistics,
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
    parser.add_argument(
        "--validation-jsonl",
        type=Path,
        help="Optional explicit validation set, kept entirely out of training.",
    )
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
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "adamw8bit"),
        default="adamw",
        help="Use adamw8bit for memory-constrained full-model SFT on CUDA.",
    )
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
    validate_chat_tokenizer(tokenizer)
    if project_config.model.vocab_size != CHAT_VOCAB_SIZE:
        raise ValueError(
            f"SFT model vocabulary must be {CHAT_VOCAB_SIZE}."
        )
    tokenizer_checksum = file_sha256(arguments.tokenizer)
    records, training_load_statistics = load_chat_records_with_statistics(
        [arguments.instruction_jsonl]
    )
    if not records:
        raise ValueError("Instruction dataset contains no valid conversations.")
    validation_path = getattr(arguments, "validation_jsonl", None)
    if validation_path is None:
        training_records, validation_records = split_instruction_records(
            records,
            validation_ratio=arguments.validation_ratio,
            seed=training_config.seed,
        )
    else:
        training_records = records
        validation_records, validation_load_statistics = (
            load_chat_records_with_statistics([validation_path])
        )
        if not validation_records:
            raise ValueError("Validation dataset contains no valid conversations.")
        training_identities = {
            format_chat_messages(record.messages) for record in training_records
        }
        validation_identities = {
            format_chat_messages(record.messages) for record in validation_records
        }
        if training_identities & validation_identities:
            raise ValueError("Training and validation sets contain duplicate conversations.")
    if validation_path is None:
        validation_load_statistics = None
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
    instruction_input_tokens = sum(
        len(input_ids) for input_ids, _labels in train_dataset.examples
    )
    supervised_target_tokens = sum(
        int((labels != -100).sum().item())
        for _input_ids, labels in train_dataset.examples
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
        optimizer_name=getattr(arguments, "optimizer", "adamw"),
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
        if (
            loaded.training_stage != "supervised_fine_tuning"
            or loaded.chat_template_version != CHAT_TEMPLATE_VERSION
        ):
            raise ValueError("Resume checkpoint uses an incompatible chat protocol.")
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
        verify_checkpoint_checksum(arguments.base_checkpoint)
        base_payload = torch.load(
            arguments.base_checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if not isinstance(base_payload, dict):
            raise ValueError("Base checkpoint payload must be an object.")
        base_config_value = base_payload.get("config")
        if not isinstance(base_config_value, dict) or not isinstance(
            base_config_value.get("model"), dict
        ):
            raise ValueError("Base checkpoint model configuration is missing.")
        base_model_config = ModelConfig(**base_config_value["model"])
        expected_geometry = asdict(project_config.model)
        actual_geometry = asdict(base_model_config)
        expected_geometry.pop("vocab_size")
        actual_geometry.pop("vocab_size")
        if actual_geometry != expected_geometry or base_model_config.vocab_size != 8192:
            raise ValueError(
                "Base checkpoint geometry must match SFT except for the "
                "8,192-to-8,196 vocabulary expansion."
            )
        base_model = LanguageModel(base_model_config).to(device)
        base = load_model_checkpoint(
            arguments.base_checkpoint,
            model=base_model,
            map_location=device,
        )
        manifest_path = arguments.tokenizer.with_name("tokenizer_manifest.json")
        if not manifest_path.is_file():
            raise ValueError("Extended chat tokenizer manifest is required.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("chat_template_version") != CHAT_TEMPLATE_VERSION:
            raise ValueError("Extended tokenizer manifest has the wrong template.")
        if base.tokenizer_sha256 is not None and manifest.get(
            "base_tokenizer_sha256"
        ) != base.tokenizer_sha256:
            raise ValueError("Base checkpoint tokenizer lineage does not match SFT.")
        base_model.resize_token_embeddings(
            CHAT_VOCAB_SIZE,
            seed=training_config.seed,
        )
        model.load_state_dict(base_model.state_dict(), strict=True)
        del base_model
        base_checkpoint_metadata = {
            "path": str(arguments.base_checkpoint),
            "run_name": base.run_name,
            "run_id": base.run_id,
            "optimizer_step": base.training_state.optimizer_step,
            "pretraining_tokens_seen": base.training_state.tokens_seen,
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
        payload["chat_special_token_ids"] = chat_special_token_map()
        payload["base_checkpoint"] = base_checkpoint_metadata
        payload["instruction_input_tokens"] = instruction_input_tokens
        payload["supervised_target_tokens"] = current_state.tokens_seen
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
                    "validation_jsonl": (
                        None if validation_path is None else str(validation_path)
                    ),
                    "validation_sha256": (
                        None if validation_path is None else file_sha256(validation_path)
                    ),
                    "instruction_record_count": len(records),
                    "instruction_load_statistics": (
                        training_load_statistics.to_dict()
                    ),
                    "training_record_count": len(training_records),
                    "validation_record_count": len(validation_records),
                    "validation_load_statistics": (
                        None
                        if validation_load_statistics is None
                        else validation_load_statistics.to_dict()
                    ),
                    "overlength_training_records_removed": (
                        train_dataset.removed_overlength
                    ),
                    "overlength_validation_records_removed": (
                        validation_dataset.removed_overlength
                    ),
                    "validation_ratio": arguments.validation_ratio,
                    "tokenizer": str(arguments.tokenizer),
                    "tokenizer_sha256": tokenizer_checksum,
                    "base_checkpoint": base_checkpoint_metadata,
                    "instruction_input_tokens": instruction_input_tokens,
                    "supervised_target_tokens_per_epoch": (
                        supervised_target_tokens
                    ),
                    "chat_special_token_ids": chat_special_token_map(),
                    "device": str(device),
                    "precision": precision.name,
                    "optimizer": getattr(arguments, "optimizer", "adamw"),
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
