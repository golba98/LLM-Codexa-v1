"""Train a byte-level BPE tokenizer from cleaned JSONL data."""

import argparse
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tokenizer import train_tokenizer, train_tokenizer_streaming


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the tokenizer-training command-line parser."""

    parser = argparse.ArgumentParser(
        description="Train a reproducible byte-level BPE tokenizer."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Cleaned train/validation JSONL files in stable order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream JSONL documents instead of retaining the corpus.",
    )
    return parser


def main() -> None:
    """Train a tokenizer from command-line arguments."""

    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        trainer = (
            train_tokenizer_streaming
            if arguments.streaming
            else train_tokenizer
        )
        result = trainer(
            arguments.inputs,
            output_dir=arguments.output_dir,
            vocab_size=arguments.vocab_size,
            min_frequency=arguments.min_frequency,
        )
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))

    print(f"Tokenizer saved: {result.tokenizer_path}")
    print(f"Manifest saved: {result.manifest_path}")
    print(f"Requested vocabulary size: {result.requested_vocab_size}")
    print(f"Actual vocabulary size: {result.actual_vocab_size}")
    print(
        "Average characters per token: "
        f"{result.inspection.average_characters_per_token:.6f}"
    )
    print(
        f"Unknown-token rate: {result.inspection.unknown_token_rate:.6f}"
    )


if __name__ == "__main__":
    main()
