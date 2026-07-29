"""Compare trusted checkpoint-evaluation reports and select a best candidate."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _finite_number(value: object, field: str, path: Path) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{path}: {field} must be a finite number.")
    return float(value)


def _load_summary(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: evaluation report must be an object.")
    step = value.get("checkpoint_optimizer_step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(
            f"{path}: checkpoint_optimizer_step must be non-negative."
        )
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path}: samples must be a non-empty array.")

    category_metrics: dict[str, dict[str, object]] = {}
    repetition_rates: list[float] = []
    malformed_count = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"{path}: sample {index} must be an object.")
        category = sample.get("category")
        quality = sample.get("quality")
        if not isinstance(category, str) or not category:
            raise ValueError(f"{path}: sample {index} has an invalid category.")
        if not isinstance(quality, dict):
            raise ValueError(f"{path}: sample {index} has invalid quality.")
        repetition = _finite_number(
            quality.get("repeated_four_gram_rate"),
            f"samples[{index}].quality.repeated_four_gram_rate",
            path,
        )
        malformed = quality.get("malformed_character_count")
        if (
            not isinstance(malformed, int)
            or isinstance(malformed, bool)
            or malformed < 0
        ):
            raise ValueError(
                f"{path}: sample {index} malformed count must be non-negative."
            )
        repetition_rates.append(repetition)
        malformed_count += malformed
        category_metrics[category] = {
            "repeated_four_gram_rate": repetition,
            "malformed_character_count": malformed,
            "word_count": quality.get("word_count"),
        }

    validation_loss = value.get("validation_loss")
    if validation_loss is not None:
        validation_loss = _finite_number(
            validation_loss,
            "validation_loss",
            path,
        )
    return {
        "report": str(path),
        "checkpoint": value.get("checkpoint"),
        "checkpoint_run_id": value.get("checkpoint_run_id"),
        "checkpoint_optimizer_step": step,
        "validation_loss": validation_loss,
        "validation_perplexity": value.get("validation_perplexity"),
        "mean_repeated_four_gram_rate": (
            sum(repetition_rates) / len(repetition_rates)
        ),
        "malformed_character_count": malformed_count,
        "category_metrics": category_metrics,
    }


def _atomic_json(path: Path, value: object, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite comparison: {path}")
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
    if len(arguments.reports) < 2:
        raise ValueError("At least two evaluation reports are required.")
    candidates = [_load_summary(path) for path in arguments.reports]
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["validation_loss"] is None,
            (
                math.inf
                if item["validation_loss"] is None
                else item["validation_loss"]
            ),
            item["mean_repeated_four_gram_rate"],
            item["malformed_character_count"],
            -item["checkpoint_optimizer_step"],
        ),
    )
    comparison = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "lowest finite validation loss, then repetition, malformed "
            "characters, and later optimizer step"
        ),
        "candidate_count": len(ranked),
        "best_checkpoint": ranked[0]["checkpoint"],
        "best_report": ranked[0]["report"],
        "candidates": ranked,
    }
    _atomic_json(arguments.output, comparison, overwrite=arguments.overwrite)
    print(f"Candidates: {len(ranked)}")
    print(f"Best checkpoint: {comparison['best_checkpoint']}")
    print(f"Comparison: {arguments.output}")
    return comparison


def main() -> None:
    run(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
