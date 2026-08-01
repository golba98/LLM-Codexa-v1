"""Append Codexa chat-control tokens to an existing 8,192-token tokenizer."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chat_protocol import (
    CHAT_TEMPLATE_VERSION,
    chat_special_token_map,
    extend_tokenizer_for_chat,
)
from src.token_data import file_sha256
from src.tokenizer import load_tokenizer


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    tokenizer = extend_tokenizer_for_chat(load_tokenizer(arguments.input))
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = arguments.output_dir / "tokenizer.json"
    manifest_path = arguments.output_dir / "tokenizer_manifest.json"
    tokenizer.save(str(tokenizer_path), pretty=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format_version": "2.0",
        "chat_template_version": CHAT_TEMPLATE_VERSION,
        "base_tokenizer": str(arguments.input),
        "base_tokenizer_sha256": file_sha256(arguments.input),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "actual_vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "special_tokens": chat_special_token_map(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    print(json.dumps(run(build_argument_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
