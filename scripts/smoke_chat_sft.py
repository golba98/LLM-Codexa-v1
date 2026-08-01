"""Train and serve a tiny real chat SFT checkpoint in a temporary directory."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_sft import run as run_sft
from src.chat_protocol import (
    CHAT_TEMPLATE_VERSION,
    chat_special_token_map,
    extend_tokenizer_for_chat,
)
from src.checkpointing import (
    CheckpointManager,
    SchedulerState,
    build_checkpoint_payload,
)
from src.config import ProjectConfig, load_config
from src.model import LanguageModel
from src.openai_server import CodexaCompletionEngine
from src.token_data import file_sha256
from src.tokenizer import load_tokenizer, train_tokenizer
from src.training import TrainingState, create_adamw_optimizer


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=240)
    return parser


def _tokenizer(root: Path) -> tuple[Path, Path]:
    corpus = root / "corpus.jsonl"
    texts: list[str] = []
    for fixture in (
        Path("tests/fixtures/chat_sft_train.jsonl"),
        Path("tests/fixtures/chat_sft_validation.jsonl"),
    ):
        for line in fixture.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            texts.extend(message["content"] for message in value["messages"])
    corpus.write_text(
        "".join(
            json.dumps({"text": text, "source": "chat-smoke"}) + "\n"
            for text in texts
        ),
        encoding="utf-8",
    )
    result = train_tokenizer(
        [corpus],
        output_dir=root / "trained-tokenizer",
        vocab_size=512,
        min_frequency=1,
    )
    tokenizer = load_tokenizer(result.tokenizer_path)
    size = tokenizer.get_vocab_size(with_added_tokens=True)
    tokenizer.add_tokens(
        [f"<unused_{index:05d}>" for index in range(8192 - size)]
    )
    base_path = root / "tokenizer-base.json"
    tokenizer.save(str(base_path))
    extend_tokenizer_for_chat(tokenizer)
    chat_directory = root / "tokenizer-chat-v3"
    chat_directory.mkdir()
    chat_path = chat_directory / "tokenizer.json"
    tokenizer.save(str(chat_path))
    manifest = {
        "format_version": "2.0",
        "chat_template_version": CHAT_TEMPLATE_VERSION,
        "base_tokenizer_sha256": file_sha256(base_path),
        "tokenizer_sha256": file_sha256(chat_path),
        "actual_vocab_size": tokenizer.get_vocab_size(True),
        "special_tokens": chat_special_token_map(),
    }
    (chat_directory / "tokenizer_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return base_path, chat_path


def run(max_steps: int) -> dict[str, object]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    with tempfile.TemporaryDirectory(prefix="codexa-chat-smoke-") as temporary:
        root = Path(temporary)
        base_tokenizer, chat_tokenizer = _tokenizer(root)
        chat_config = load_config("configs/smoke_chat_sft.yaml")
        base_model_config = replace(chat_config.model, vocab_size=8192)
        base_project_config = ProjectConfig(
            model=base_model_config,
            training=chat_config.training,
        )
        base_model = LanguageModel(base_model_config)
        optimizer = create_adamw_optimizer(
            base_model,
            learning_rate=chat_config.training.learning_rate,
            weight_decay=chat_config.training.weight_decay,
        )
        base_checkpoint = CheckpointManager(root / "checkpoints", "base").save(
            build_checkpoint_payload(
                model=base_model,
                optimizer=optimizer,
                scaler=None,
                state=TrainingState(),
                scheduler=SchedulerState(
                    warmup_steps=chat_config.training.warmup_steps,
                    max_steps=chat_config.training.max_steps,
                    peak_learning_rate=chat_config.training.learning_rate,
                    minimum_learning_rate=chat_config.training.learning_rate * 0.1,
                ),
                config=base_project_config,
                run_name="chat-smoke-base",
                run_id="chat-smoke-base",
                tokenizer_reference=str(base_tokenizer),
                tokenizer_sha256=file_sha256(base_tokenizer),
            )
        )
        del optimizer, base_model
        result = run_sft(
            SimpleNamespace(
                config=Path("configs/smoke_chat_sft.yaml"),
                base_checkpoint=base_checkpoint,
                resume=None,
                tokenizer=chat_tokenizer,
                instruction_jsonl=Path("tests/fixtures/chat_sft_train.jsonl"),
                validation_jsonl=Path(
                    "tests/fixtures/chat_sft_validation.jsonl"
                ),
                validation_ratio=0.05,
                device="cpu",
                precision="fp32",
                max_steps=max_steps,
                max_validation_batches=8,
                num_workers=0,
                log_dir=root / "logs",
                run_name="chat-smoke-sft",
                checkpoint_dir=root / "checkpoints",
                overwrite_log=True,
            )
        )
        if result != 0:
            raise RuntimeError(f"SFT smoke returned {result}.")
        checkpoint = root / "checkpoints/chat-smoke-sft/latest.pt"
        engine = CodexaCompletionEngine(
            checkpoint,
            chat_tokenizer,
            device="cpu",
            precision="fp32",
        )
        cases = {
            "hi": "Hi!",
            "Just say hi.": "Hi.",
            "What is 2 + 2?": "4",
        }
        outputs: dict[str, str] = {}
        for prompt, expected in cases.items():
            completion = engine.complete(
                [{"role": "user", "content": prompt}],
                max_tokens=16,
                temperature=None,
                top_p=None,
                repetition_penalty=1.0,
                seed=42,
            )
            outputs[prompt] = completion.text
            if completion.text != expected or completion.finish_reason != "stop":
                raise AssertionError(
                    f"Smoke prompt {prompt!r} produced {completion.text!r} "
                    f"with {completion.finish_reason!r}."
                )
        metrics_path = root / "logs/chat-smoke-sft/train_metrics.jsonl"
        metrics = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
        ]
        final = metrics[-1]
        return {
            "max_steps": max_steps,
            "outputs": outputs,
            "validation_loss": final.get("validation_loss"),
            "validation_perplexity": final.get("validation_perplexity"),
            "supervised_target_tokens_seen": final["total_tokens_seen"],
            "status": "passed",
        }


def main() -> None:
    arguments = build_argument_parser().parse_args()
    print(json.dumps(run(arguments.max_steps), indent=2))


if __name__ == "__main__":
    main()
