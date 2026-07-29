"""Tests for instruction formatting and response-masked SFT data."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from scripts.train_sft import run as run_sft
from src.checkpointing import (
    CheckpointManager,
    SchedulerState,
    build_checkpoint_payload,
)
from src.config import ProjectConfig, TrainingConfig
from src.model import LanguageModel, ModelConfig
from src.sft import (
    IGNORE_INDEX,
    InstructionDataset,
    format_instruction_prompt,
    instruction_collate,
    load_instruction_records,
    split_instruction_records,
)
from src.tokenizer import load_tokenizer, train_tokenizer
from src.token_data import file_sha256
from src.training import (
    TrainingState,
    create_adamw_optimizer,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        records_path = root / "instructions.jsonl"
        values = [
            {
                "instruction": "Name one color.",
                "context": "",
                "response": "Blue.",
                "category": "closed_qa",
            },
            {
                "instruction": "Add the numbers.",
                "context": "Two and three.",
                "response": "Five.",
                "category": "closed_qa",
            },
            {
                "instruction": "Greet the reader.",
                "context": "",
                "response": "Hello!",
                "category": "creative_writing",
            },
        ]
        records_path.write_text(
            "".join(json.dumps(value) + "\n" for value in values),
            encoding="utf-8",
        )
        tokenizer_corpus = root / "corpus.jsonl"
        tokenizer_corpus.write_text(
            "".join(
                json.dumps(
                    {
                        "text": (
                            format_instruction_prompt(
                                value["instruction"],
                                value["context"],
                            )
                            + " "
                            + value["response"]
                        ),
                        "source": "test",
                    }
                )
                + "\n"
                for value in values
            ),
            encoding="utf-8",
        )
        tokenizer = train_tokenizer(
            [tokenizer_corpus],
            output_dir=root / "tokenizer",
            vocab_size=300,
            min_frequency=1,
        )
        records = load_instruction_records(records_path)
        first_split = split_instruction_records(
            records,
            validation_ratio=0.34,
            seed=42,
        )
        second_split = split_instruction_records(
            records,
            validation_ratio=0.34,
            seed=42,
        )
        assert first_split == second_split
        assert sum(map(len, first_split)) == 3
        dataset = InstructionDataset(
            records,
            tokenizer=load_tokenizer(tokenizer.tokenizer_path),
            context_length=32,
        )
        input_ids, labels = instruction_collate(
            [dataset[0], dataset[1]]
        )
        assert input_ids.shape == labels.shape
        assert input_ids.ndim == 2
        assert (labels == IGNORE_INDEX).any()
        assert (labels != IGNORE_INDEX).any()
        assert torch.equal(
            input_ids[labels != IGNORE_INDEX],
            labels[labels != IGNORE_INDEX],
        )

        model_config = ModelConfig(
            vocab_size=512,
            context_length=32,
            num_layers=1,
            hidden_size=16,
            num_heads=4,
            intermediate_size=32,
        )
        base_training = TrainingConfig(
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_steps=0,
            max_steps=2,
            gradient_clip=1.0,
            precision="fp32",
            checkpoint_interval=1,
            evaluation_interval=1,
            seed=42,
        )
        project_config = ProjectConfig(model_config, base_training)
        model = LanguageModel(model_config)
        optimizer = create_adamw_optimizer(
            model,
            learning_rate=base_training.learning_rate,
            weight_decay=base_training.weight_decay,
        )
        base_checkpoint = CheckpointManager(
            root / "base-checkpoints",
            "base",
        ).save(
            build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scaler=None,
                state=TrainingState(),
                scheduler=SchedulerState(
                    warmup_steps=0,
                    max_steps=2,
                    peak_learning_rate=1e-3,
                    minimum_learning_rate=1e-4,
                ),
                config=project_config,
                run_name="base",
                run_id="base-run",
                tokenizer_reference=str(tokenizer.tokenizer_path),
                tokenizer_sha256=file_sha256(tokenizer.tokenizer_path),
            )
        )
        config_path = root / "sft.yaml"
        config_path.write_text(
            """
model:
  vocab_size: 512
  context_length: 32
  num_layers: 1
  hidden_size: 16
  num_heads: 4
  intermediate_size: 32
  dropout: 0.0
  tie_embeddings: true
training:
  micro_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 0.001
  weight_decay: 0.0
  warmup_steps: 0
  max_steps: 2
  gradient_clip: 1.0
  precision: fp32
  checkpoint_interval: 1
  evaluation_interval: 1
  seed: 42
""".lstrip(),
            encoding="utf-8",
        )
        result = run_sft(
            SimpleNamespace(
                config=config_path,
                base_checkpoint=base_checkpoint,
                resume=None,
                tokenizer=tokenizer.tokenizer_path,
                instruction_jsonl=records_path,
                validation_ratio=0.34,
                device="cpu",
                precision="fp32",
                max_steps=1,
                max_validation_batches=1,
                num_workers=0,
                log_dir=root / "logs",
                run_name="sft-smoke",
                checkpoint_dir=root / "sft-checkpoints",
                overwrite_log=True,
            )
        )
        assert result == 0
        assert (root / "sft-checkpoints/sft-smoke/latest.pt").is_file()
        metadata = json.loads(
            (root / "logs/sft-smoke/run_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["training_stage"] == "supervised_fine_tuning"
        assert metadata["base_checkpoint"]["run_id"] == "base-run"

    print("All SFT dataset tests passed.")


if __name__ == "__main__":
    main()
