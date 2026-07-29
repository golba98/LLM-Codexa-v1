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
    CHAT_TEMPLATE_VERSION,
    ChatMessage,
    ChatRecord,
    IGNORE_INDEX,
    InstructionDataset,
    format_chat_messages,
    format_instruction_prompt,
    instruction_collate,
    load_chat_records,
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
    assert CHAT_TEMPLATE_VERSION == "2.0"
    conversation = (
        ChatMessage("system", "Be concise."),
        ChatMessage("user", "Remember the color blue."),
        ChatMessage("assistant", "I will remember blue."),
        ChatMessage("user", "Which color did I name?"),
        ChatMessage("assistant", "Blue."),
    )
    assert format_chat_messages(conversation) == (
        "System: Be concise.\n"
        "User: Remember the color blue.\n"
        "Assistant: I will remember blue.\n"
        "User: Which color did I name?\n"
        "Assistant: Blue."
    )
    assert format_chat_messages(
        conversation[:-1],
        add_generation_prompt=True,
    ).endswith("User: Which color did I name?\nAssistant:")

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
        legacy_chat_records = load_chat_records(records_path)
        assert len(legacy_chat_records) == len(records)
        assert legacy_chat_records[0].messages[0].role == "user"
        assert legacy_chat_records[0].messages[-1].role == "assistant"

        chat_path = root / "chat.jsonl"
        chat_value = {
            "messages": [
                {"role": message.role, "content": message.content}
                for message in conversation
            ],
            "category": "context_retention",
            "source": "original-test",
            "conversation_id": "chat-1",
        }
        chat_path.write_text(
            json.dumps(chat_value)
            + "\n"
            + json.dumps(chat_value)
            + "\n",
            encoding="utf-8",
        )
        chat_records = load_chat_records(chat_path)
        assert len(chat_records) == 1
        assert chat_records[0].conversation_id == "chat-1"
        assert len(chat_records[0].messages) == 5

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
        assert input_ids[0, 0].item() == 1
        assert labels[0, 0].item() == IGNORE_INDEX
        first_targets = labels[0][labels[0] != IGNORE_INDEX].tolist()
        assert first_targets[-1] == 2
        assert "Blue" in load_tokenizer(
            tokenizer.tokenizer_path
        ).decode(first_targets[:-1])

        multi_turn_dataset = InstructionDataset(
            chat_records,
            tokenizer=load_tokenizer(tokenizer.tokenizer_path),
            context_length=256,
        )
        multi_input, multi_labels = multi_turn_dataset[0]
        assert multi_input.shape == multi_labels.shape
        target_ids = multi_labels[multi_labels != IGNORE_INDEX].tolist()
        target_text = load_tokenizer(tokenizer.tokenizer_path).decode(
            target_ids[:-1]
        )
        assert "remember" in target_text
        assert "Blue" in target_text
        assert target_ids[-1] == 2

        padded_inputs, padded_labels = instruction_collate(
            [multi_turn_dataset[0], dataset[0]]
        )
        shorter_length = len(dataset[0][0])
        assert torch.all(
            padded_labels[1, shorter_length:] == IGNORE_INDEX
        )
        assert torch.all(padded_inputs[1, shorter_length:] == 0)

        malformed_path = root / "malformed-chat.jsonl"
        malformed_path.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "assistant", "content": "Wrong order."},
                        {"role": "user", "content": "Hello."},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            load_chat_records(malformed_path)
        except ValueError as error:
            assert "must be user" in str(error)
        else:
            raise AssertionError("Malformed chat role order was accepted.")

        model_config = ModelConfig(
            vocab_size=512,
            context_length=32,
            num_layers=1,
            hidden_size=16,
            num_heads=4,
            intermediate_size=32,
        )
        overfit_model = LanguageModel(model_config)
        overfit_optimizer = torch.optim.AdamW(
            overfit_model.parameters(),
            lr=0.02,
            weight_decay=0.0,
        )
        overfit_inputs, overfit_labels = instruction_collate([dataset[0]])
        with torch.no_grad():
            _, initial_overfit_loss = overfit_model(
                overfit_inputs,
                overfit_labels,
            )
        assert initial_overfit_loss is not None
        for _ in range(50):
            overfit_optimizer.zero_grad(set_to_none=True)
            _, overfit_loss = overfit_model(overfit_inputs, overfit_labels)
            assert overfit_loss is not None and torch.isfinite(overfit_loss)
            overfit_loss.backward()
            overfit_optimizer.step()
        with torch.no_grad():
            _, final_overfit_loss = overfit_model(
                overfit_inputs,
                overfit_labels,
            )
        assert final_overfit_loss is not None
        assert final_overfit_loss < initial_overfit_loss * 0.2

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
