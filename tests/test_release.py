"""End-to-end tests for exported inference release bundles."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from scripts.export_release import run as export_release
from src.checkpointing import (
    CheckpointManager,
    SchedulerState,
    build_checkpoint_payload,
)
from src.config import ProjectConfig, TrainingConfig
from src.generate import GenerationConfig, generate_token_ids
from src.model import LanguageModel, ModelConfig, count_parameters
from src.release import load_release, verify_release_directory
from src.token_data import file_sha256
from src.tokenizer import EOS_TOKEN, train_tokenizer
from src.training import TrainingState, create_adamw_optimizer


def _raises(exception_type: type[BaseException], operation) -> None:
    try:
        operation()
    except exception_type:
        return
    raise AssertionError(f"Expected {exception_type.__name__}.")


def _config() -> ProjectConfig:
    return ProjectConfig(
        model=ModelConfig(
            vocab_size=260,
            context_length=16,
            num_layers=1,
            hidden_size=16,
            num_heads=4,
            intermediate_size=32,
            dropout=0.0,
        ),
        training=TrainingConfig(
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            weight_decay=0.01,
            warmup_steps=1,
            max_steps=4,
            gradient_clip=1.0,
            precision="fp32",
            checkpoint_interval=1,
            evaluation_interval=1,
            seed=42,
        ),
    )


def _rewrite_integrity_files(release_dir: Path) -> None:
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["training_state.json"] = file_sha256(
        release_dir / "training_state.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksums = {
        **manifest["artifacts"],
        "release_manifest.json": file_sha256(manifest_path),
    }
    (release_dir / "SHA256SUMS").write_text(
        "".join(
            f"{checksum}  {filename}\n"
            for filename, checksum in sorted(checksums.items())
        ),
        encoding="utf-8",
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        corpus = root / "corpus.jsonl"
        corpus.write_text(
            json.dumps(
                {
                    "text": (
                        "A small original corpus exercises release export, "
                        "loading, checksum verification, and generation."
                    )
                }
            )
            + "\n",
            encoding="utf-8",
        )
        tokenizer_result = train_tokenizer(
            [corpus],
            output_dir=root / "tokenizer",
            vocab_size=260,
            min_frequency=1,
        )
        tokenizer_checksum = file_sha256(tokenizer_result.tokenizer_path)

        config = _config()
        model = LanguageModel(config.model)
        optimizer = create_adamw_optimizer(
            model,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        state = TrainingState(
            micro_step=3,
            optimizer_step=3,
            tokens_seen=48,
            completed_epochs=1,
        )
        payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=None,
            state=state,
            scheduler=SchedulerState(
                warmup_steps=1,
                max_steps=4,
                peak_learning_rate=1e-3,
                minimum_learning_rate=1e-4,
            ),
            config=config,
            run_name="release-test",
            run_id="release-test-id",
            tokenizer_reference=str(tokenizer_result.tokenizer_path),
            tokenizer_sha256=tokenizer_checksum,
        )
        checkpoint = CheckpointManager(
            root / "checkpoints",
            "release-test",
        ).save(payload, is_best=True, milestone=True)

        release_dir = root / "release"
        export_release(
            SimpleNamespace(
                checkpoint=checkpoint,
                tokenizer=tokenizer_result.tokenizer_path,
                output_dir=release_dir,
            )
        )
        manifest = verify_release_directory(release_dir)
        assert manifest["source_checkpoint_optimizer_step"] == 3
        assert manifest["tokenizer_sha256"] == tokenizer_checksum
        assert manifest["parameter_count"] == count_parameters(model)
        assert (release_dir / "model.safetensors").is_file()
        assert (release_dir / "tokenizer.json").is_file()
        assert (release_dir / "SHA256SUMS").is_file()

        bundle = load_release(release_dir, device="cpu")
        assert bundle.model.training is False
        for expected, loaded in zip(
            model.parameters(),
            bundle.model.parameters(),
            strict=True,
        ):
            assert torch.equal(expected, loaded)
        generated = generate_token_ids(
            bundle.model,
            torch.tensor([[1]], dtype=torch.long),
            eos_token_id=bundle.tokenizer.token_to_id(EOS_TOKEN),
            config=GenerationConfig(
                max_new_tokens=2,
                do_sample=False,
            ),
        )
        assert generated.shape[0] == 1
        assert 1 <= generated.shape[1] <= 3

        _raises(
            FileExistsError,
            lambda: export_release(
                SimpleNamespace(
                    checkpoint=checkpoint,
                    tokenizer=tokenizer_result.tokenizer_path,
                    output_dir=release_dir,
                )
            ),
        )

        training_state_path = release_dir / "training_state.json"
        corrupted_state = json.loads(
            training_state_path.read_text(encoding="utf-8")
        )
        corrupted_state["optimizer_step"] = 999
        training_state_path.write_text(
            json.dumps(corrupted_state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _rewrite_integrity_files(release_dir)
        _raises(
            ValueError,
            lambda: verify_release_directory(release_dir),
        )

    print("All release-export tests passed.")


if __name__ == "__main__":
    main()
