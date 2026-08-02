"""Generate a base-model text continuation from a trusted checkpoint."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import torch


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.checkpointing import load_model_checkpoint, verify_checkpoint_checksum
from src.generate import GenerationConfig, generate_sequences
from src.model import LanguageModel, ModelConfig
from src.token_data import file_sha256
from src.tokenizer import BOS_TOKEN, EOS_TOKEN, load_tokenizer
from src.training import resolve_device


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the base-generation command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _checkpoint_model_config(path: Path) -> ModelConfig:
    """Read the model geometry after checksum validation."""

    verify_checkpoint_checksum(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be an object.")
    config = payload.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Checkpoint model configuration is missing.")
    try:
        return ModelConfig(**config["model"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid checkpoint model configuration: {error}") from error


def _atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(
                value,
                output_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def run(arguments: argparse.Namespace) -> dict[str, object]:
    """Generate and optionally save one continuation."""

    device = resolve_device(arguments.device)
    model = LanguageModel(_checkpoint_model_config(arguments.checkpoint)).to(device)
    checkpoint = load_model_checkpoint(
        arguments.checkpoint,
        model=model,
        map_location=device,
    )
    tokenizer = load_tokenizer(arguments.tokenizer)
    tokenizer_checksum = file_sha256(arguments.tokenizer)
    if (
        checkpoint.tokenizer_sha256 is not None
        and checkpoint.tokenizer_sha256 != tokenizer_checksum
    ):
        raise ValueError("Tokenizer checksum does not match the checkpoint.")

    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
    bos_token_id = tokenizer.token_to_id(BOS_TOKEN)
    if eos_token_id != 2 or bos_token_id != 1:
        raise ValueError("Tokenizer must use <bos>=1 and <eos>=2.")
    prompt_ids = tokenizer.encode(
        arguments.prompt,
        add_special_tokens=False,
    ).ids or [bos_token_id]

    generation_config = GenerationConfig(
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_k=arguments.top_k,
        top_p=arguments.top_p,
        repetition_penalty=arguments.repetition_penalty,
        do_sample=not arguments.greedy,
        seed=arguments.seed,
    )
    generated = generate_sequences(
        model,
        torch.tensor([prompt_ids], dtype=torch.long, device=device),
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.token_to_id("<pad>"),
        config=generation_config,
    ).sequences[0]
    continuation_ids = list(generated.visible_token_ids)
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_run_id": checkpoint.run_id,
        "checkpoint_optimizer_step": checkpoint.training_state.optimizer_step,
        "tokenizer": str(arguments.tokenizer),
        "tokenizer_sha256": tokenizer_checksum,
        "device": str(device),
        "prompt": arguments.prompt,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": continuation_ids,
        "generated_token_count": generated.generated_token_count,
        "finish_reason": generated.finish_reason,
        "termination_cause": generated.termination_cause,
        "terminating_token_id": generated.terminating_token_id,
        "text": tokenizer.decode(continuation_ids, skip_special_tokens=True),
        "generation_config": asdict(generation_config),
    }
    if arguments.output is not None:
        _atomic_json_write(arguments.output, output)
    print(output["text"])
    return output


def main() -> None:
    run(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
