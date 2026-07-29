"""Evaluate one Codexa checkpoint with validation loss and fixed prompts."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import torch


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate import _checkpoint_model_config
from src.checkpointing import load_model_checkpoint, verify_checkpoint_checksum
from src.evaluation import (
    analyze_generated_text,
    ngram_overlap_rate,
    perplexity_from_loss,
)
from src.generate import GenerationConfig, generate_token_ids
from src.model import LanguageModel, count_parameters
from src.token_data import (
    MemmapTokenDataset,
    create_token_dataloader,
    file_sha256,
)
from src.tokenizer import BOS_TOKEN, EOS_TOKEN, load_tokenizer
from src.training import evaluate, resolve_device, resolve_precision


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("configs/evaluation_prompts.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--validation-token-file", type=Path)
    parser.add_argument("--token-manifest", type=Path)
    parser.add_argument("--max-validation-batches", type=int, default=16)
    parser.add_argument("--reference-jsonl", type=Path)
    parser.add_argument("--max-reference-documents", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_prompts(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("Prompt suite must be a non-empty JSON array.")
    prompts: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or not isinstance(
            item.get("category"), str
        ):
            raise ValueError(f"Invalid prompt suite entry at index {index}.")
        prompt = item.get("prompt")
        target_tokens = item.get("target_prompt_tokens")
        simple = isinstance(prompt, str) and bool(prompt)
        long_context = (
            isinstance(target_tokens, int)
            and not isinstance(target_tokens, bool)
            and target_tokens > 0
            and all(
                isinstance(item.get(key), str) and bool(item[key])
                for key in ("prompt_prefix", "prompt_filler", "prompt_suffix")
            )
        )
        if simple == long_context:
            raise ValueError(
                f"Prompt suite entry {index} must define exactly one prompt "
                "form."
            )
        expected_terms = item.get("expected_terms", [])
        if (
            not isinstance(expected_terms, list)
            or any(
                not isinstance(term, str) or not term
                for term in expected_terms
            )
        ):
            raise ValueError(
                f"Prompt suite entry {index} has invalid expected_terms."
            )
        expected_pattern = item.get("expected_pattern")
        if expected_pattern is not None:
            if not isinstance(expected_pattern, str) or not expected_pattern:
                raise ValueError(
                    f"Prompt suite entry {index} has invalid expected_pattern."
                )
            try:
                re.compile(expected_pattern)
            except re.error as error:
                raise ValueError(
                    f"Prompt suite entry {index} has invalid expected_pattern."
                ) from error
        prompts.append(dict(item))
    return prompts


def _build_prompt(
    entry: dict[str, object],
    *,
    tokenizer,
    context_length: int,
) -> tuple[str, list[int]]:
    prompt = entry.get("prompt")
    if isinstance(prompt, str):
        return prompt, tokenizer.encode(
            prompt,
            add_special_tokens=False,
        ).ids

    target_tokens = int(entry["target_prompt_tokens"])
    if target_tokens >= context_length:
        raise ValueError(
            f"Long-context target {target_tokens} must be below model "
            f"context length {context_length}."
        )
    prefix_ids = tokenizer.encode(
        str(entry["prompt_prefix"]),
        add_special_tokens=False,
    ).ids
    filler_ids = tokenizer.encode(
        str(entry["prompt_filler"]),
        add_special_tokens=False,
    ).ids
    suffix_ids = tokenizer.encode(
        str(entry["prompt_suffix"]),
        add_special_tokens=False,
    ).ids
    fixed_length = len(prefix_ids) + len(suffix_ids)
    if not filler_ids:
        raise ValueError("Long-context filler must produce at least one token.")
    if fixed_length > target_tokens:
        raise ValueError(
            "Long-context prefix and suffix exceed target_prompt_tokens."
        )
    filler_length = target_tokens - fixed_length
    repeated_filler = (
        filler_ids * (filler_length // len(filler_ids) + 1)
    )[:filler_length]
    prompt_ids = prefix_ids + repeated_filler + suffix_ids
    if len(prompt_ids) != target_tokens:
        raise RuntimeError("Long-context prompt token accounting mismatch.")
    return tokenizer.decode(prompt_ids), prompt_ids


def _reference_text(path: Path, maximum_documents: int) -> str:
    if maximum_documents <= 0:
        raise ValueError("--max-reference-documents must be positive.")
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: malformed JSON ({error.msg})"
                ) from error
            if not isinstance(record, dict) or not isinstance(
                record.get("text"), str
            ):
                raise ValueError(
                    f"{path}:{line_number}: required text must be a string."
                )
            texts.append(record["text"])
            if len(texts) >= maximum_documents:
                break
    return "\n".join(texts)


def _validate_tokenizer_compatibility(
    *,
    checkpoint_checksum: str | None,
    tokenizer_checksum: str,
    tokenizer_vocab_size: int,
    model_vocab_size: int,
) -> None:
    if (
        checkpoint_checksum is not None
        and checkpoint_checksum != tokenizer_checksum
    ):
        raise ValueError("Tokenizer checksum does not match the checkpoint.")
    if tokenizer_vocab_size > model_vocab_size:
        raise ValueError("Tokenizer vocabulary exceeds the model vocabulary.")


def _atomic_json(path: Path, value: object, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite evaluation: {path}")
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
    if arguments.max_validation_batches <= 0:
        raise ValueError("--max-validation-batches must be positive.")
    verify_checkpoint_checksum(arguments.checkpoint)
    device = resolve_device(arguments.device)
    model_config = _checkpoint_model_config(arguments.checkpoint)
    model = LanguageModel(model_config).to(device)
    checkpoint = load_model_checkpoint(
        arguments.checkpoint,
        model=model,
        map_location=device,
    )
    tokenizer = load_tokenizer(arguments.tokenizer)
    tokenizer_checksum = file_sha256(arguments.tokenizer)
    _validate_tokenizer_compatibility(
        checkpoint_checksum=checkpoint.tokenizer_sha256,
        tokenizer_checksum=tokenizer_checksum,
        tokenizer_vocab_size=tokenizer.get_vocab_size(
            with_added_tokens=True
        ),
        model_vocab_size=model_config.vocab_size,
    )
    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
    bos_token_id = tokenizer.token_to_id(BOS_TOKEN)
    if eos_token_id != 2 or bos_token_id != 1:
        raise ValueError("Tokenizer must use <bos>=1 and <eos>=2.")

    validation_loss: float | None = None
    validation_tokens = 0
    if (arguments.validation_token_file is None) != (
        arguments.token_manifest is None
    ):
        raise ValueError(
            "Validation token file and token manifest must be supplied together."
        )
    if arguments.validation_token_file is not None:
        manifest = json.loads(
            arguments.token_manifest.read_text(encoding="utf-8")
        )
        if not isinstance(manifest, dict):
            raise ValueError("Token manifest must contain a JSON object.")
        if manifest.get("context_length") != model_config.context_length:
            raise ValueError(
                "Token-data context length does not match the checkpoint."
            )
        if manifest.get("model_vocab_size") != model_config.vocab_size:
            raise ValueError(
                "Token-data model vocabulary does not match the checkpoint."
            )
        if manifest.get("tokenizer_sha256") != tokenizer_checksum:
            raise ValueError(
                "Token-data tokenizer checksum does not match the tokenizer."
            )
        output_checksums = manifest.get("output_checksums")
        if not isinstance(output_checksums, dict) or not isinstance(
            output_checksums.get("validation"), str
        ):
            raise ValueError(
                "Token manifest validation checksum is missing."
            )
        if (
            file_sha256(arguments.validation_token_file)
            != output_checksums["validation"]
        ):
            raise ValueError("Validation token-file checksum mismatch.")
        dtype = np.dtype(manifest["dtype"])
        dataset = MemmapTokenDataset(
            arguments.validation_token_file,
            dtype=dtype,
            context_length=model_config.context_length,
            model_vocab_size=model_config.vocab_size,
        )
        loader = create_token_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        precision_name = "bf16" if device.type == "cuda" else "fp32"
        validation_loss, validation_tokens = evaluate(
            model,
            loader,
            device=device,
            precision=resolve_precision(precision_name, device),
            max_batches=arguments.max_validation_batches,
            non_blocking=device.type == "cuda",
        )

    reference_text = (
        None
        if arguments.reference_jsonl is None
        else _reference_text(
            arguments.reference_jsonl,
            arguments.max_reference_documents,
        )
    )
    generation_config = GenerationConfig(
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_k=arguments.top_k,
        top_p=arguments.top_p,
        repetition_penalty=arguments.repetition_penalty,
        do_sample=True,
        seed=arguments.seed,
    )
    samples: list[dict[str, object]] = []
    for prompt_entry in _read_prompts(arguments.prompts):
        prompt, prompt_ids = _build_prompt(
            prompt_entry,
            tokenizer=tokenizer,
            context_length=model_config.context_length,
        )
        prompt_ids = prompt_ids or [bos_token_id]
        generated = generate_token_ids(
            model,
            torch.tensor([prompt_ids], dtype=torch.long, device=device),
            eos_token_id=eos_token_id,
            config=generation_config,
        )[0].tolist()
        text = tokenizer.decode(generated, skip_special_tokens=True)
        continuation = tokenizer.decode(
            generated[len(prompt_ids) :],
            skip_special_tokens=True,
        )
        expected_terms = [
            str(term) for term in prompt_entry.get("expected_terms", [])
        ]
        expected_pattern = prompt_entry.get("expected_pattern")
        sample: dict[str, object] = {
            "category": prompt_entry["category"],
            "prompt": prompt,
            "prompt_token_count": len(prompt_ids),
            "text": text,
            "continuation": continuation,
            "generated_token_count": len(generated) - len(prompt_ids),
            "expected_terms": expected_terms,
            "expected_term_matches": {
                term: term.casefold() in continuation.casefold()
                for term in expected_terms
            },
            "expected_pattern": expected_pattern,
            "expected_pattern_match": (
                None
                if expected_pattern is None
                else re.fullmatch(
                    str(expected_pattern),
                    continuation,
                    flags=re.DOTALL,
                )
                is not None
            ),
            "quality": analyze_generated_text(continuation).to_dict(),
            "reference_eight_gram_overlap": (
                None
                if reference_text is None
                else ngram_overlap_rate(
                    continuation,
                    reference_text,
                    ngram_size=8,
                )
            ),
        }
        samples.append(sample)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_optimizer_step": checkpoint.training_state.optimizer_step,
        "checkpoint_run_id": checkpoint.run_id,
        "tokenizer": str(arguments.tokenizer),
        "tokenizer_sha256": tokenizer_checksum,
        "prompt_suite": str(arguments.prompts),
        "device": str(device),
        "parameter_count": count_parameters(model),
        "validation_loss": validation_loss,
        "validation_perplexity": (
            None
            if validation_loss is None
            else perplexity_from_loss(validation_loss)
        ),
        "validation_token_count": validation_tokens,
        "generation_config": asdict(generation_config),
        "reference_document_limit": (
            None
            if arguments.reference_jsonl is None
            else arguments.max_reference_documents
        ),
        "samples": samples,
    }
    for value in (
        report["validation_loss"],
        report["validation_perplexity"],
    ):
        if value is not None and not math.isfinite(value):
            raise FloatingPointError("Evaluation produced a non-finite metric.")
    _atomic_json(arguments.output, report, overwrite=arguments.overwrite)
    print(f"Checkpoint step: {report['checkpoint_optimizer_step']}")
    print(f"Validation loss: {validation_loss}")
    print(f"Samples: {len(samples)}")
    print(f"Report: {arguments.output}")
    return report


def main() -> None:
    arguments = build_argument_parser().parse_args()
    run(arguments)


if __name__ == "__main__":
    main()
