"""Reproducible byte-level BPE tokenizer training and inspection."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from collections.abc import Iterable, Sequence

import tokenizers
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from src.data.io import TextDocument, load_documents, write_json


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
TOKENIZER_FORMAT_VERSION = "1.0"


@dataclass(frozen=True)
class TokenizerInspection:
    """Aggregate tokenization metrics for a text collection."""

    document_count: int
    total_characters: int
    total_utf8_bytes: int
    total_tokens: int
    unknown_token_count: int
    unknown_token_rate: float
    average_characters_per_token: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable inspection mapping."""

        return asdict(self)


@dataclass(frozen=True)
class TokenizerTrainingResult:
    """Paths and metrics produced by tokenizer training."""

    tokenizer_path: Path
    manifest_path: Path
    requested_vocab_size: int
    actual_vocab_size: int
    inspection: TokenizerInspection


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_training_settings(
    vocab_size: int,
    min_frequency: int,
) -> None:
    minimum_vocab_size = len(ByteLevel.alphabet()) + len(SPECIAL_TOKENS)
    if (
        not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size < minimum_vocab_size
    ):
        raise ValueError(
            f"vocab_size must be an integer of at least {minimum_vocab_size}; "
            f"got {vocab_size!r}."
        )
    if (
        not isinstance(min_frequency, int)
        or isinstance(min_frequency, bool)
        or min_frequency <= 0
    ):
        raise ValueError(
            "min_frequency must be a positive integer; "
            f"got {min_frequency!r}."
        )


def load_tokenizer_corpus(
    input_paths: Sequence[str | Path],
) -> list[TextDocument]:
    """Load cleaned JSONL/NDJSON documents in stable input order."""

    normalized_paths = [Path(path) for path in input_paths]
    if not normalized_paths:
        raise ValueError("At least one tokenizer input path is required.")
    for path in normalized_paths:
        if path.suffix.lower() not in {".jsonl", ".ndjson"}:
            raise ValueError(
                f"Tokenizer training input must be JSONL/NDJSON; got {path}."
            )

    documents = load_documents(normalized_paths)
    if not documents:
        raise ValueError("Tokenizer training corpus contains no documents.")
    if any(not document.text for document in documents):
        raise ValueError(
            "Tokenizer training corpus contains an empty document; "
            "run dataset preparation first."
        )
    return documents


def validate_tokenizer(tokenizer: Tokenizer) -> None:
    """Validate stable special-token IDs and byte-level components."""

    if not isinstance(tokenizer.model, BPE):
        raise ValueError("Tokenizer model must be BPE.")
    if not isinstance(tokenizer.pre_tokenizer, ByteLevel):
        raise ValueError("Tokenizer pre-tokenizer must be ByteLevel.")
    if not isinstance(tokenizer.decoder, ByteLevelDecoder):
        raise ValueError("Tokenizer decoder must be ByteLevel.")

    for expected_id, special_token in enumerate(SPECIAL_TOKENS):
        actual_id = tokenizer.token_to_id(special_token)
        if actual_id != expected_id:
            raise ValueError(
                f"{special_token} must have token ID {expected_id}; "
                f"got {actual_id!r}."
            )


def inspect_tokenizer(
    tokenizer: Tokenizer,
    texts: Iterable[str],
) -> TokenizerInspection:
    """Calculate round-trip and unknown-token metrics."""

    text_values = list(texts)
    unknown_id = tokenizer.token_to_id(UNK_TOKEN)
    if unknown_id is None:
        raise ValueError("Tokenizer does not contain the required <unk> token.")

    total_characters = 0
    total_utf8_bytes = 0
    total_tokens = 0
    unknown_token_count = 0
    for text in text_values:
        encoding = tokenizer.encode(text)
        total_characters += len(text)
        total_utf8_bytes += len(text.encode("utf-8"))
        total_tokens += len(encoding.ids)
        unknown_token_count += encoding.ids.count(unknown_id)

    return TokenizerInspection(
        document_count=len(text_values),
        total_characters=total_characters,
        total_utf8_bytes=total_utf8_bytes,
        total_tokens=total_tokens,
        unknown_token_count=unknown_token_count,
        unknown_token_rate=(
            unknown_token_count / total_tokens if total_tokens else 0.0
        ),
        average_characters_per_token=(
            total_characters / total_tokens if total_tokens else 0.0
        ),
    )


def train_tokenizer(
    input_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    vocab_size: int = 8192,
    min_frequency: int = 2,
) -> TokenizerTrainingResult:
    """Train and save a deterministic byte-level BPE tokenizer."""

    _validate_training_settings(vocab_size, min_frequency)
    normalized_paths = [Path(path) for path in input_paths]
    documents = load_tokenizer_corpus(normalized_paths)
    texts = [document.text for document in documents]

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(
        texts,
        trainer=trainer,
        length=len(texts),
    )
    validate_tokenizer(tokenizer)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer_path = destination / "tokenizer.json"
    manifest_path = destination / "tokenizer_manifest.json"
    tokenizer.save(str(tokenizer_path), pretty=True)

    inspection = inspect_tokenizer(tokenizer, texts)
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "format_version": TOKENIZER_FORMAT_VERSION,
        "tokenizers_version": tokenizers.__version__,
        "input_paths": [str(path.resolve()) for path in normalized_paths],
        "input_sha256": {
            str(path.resolve()): _file_sha256(path) for path in normalized_paths
        },
        "model_type": "BPE",
        "pre_tokenizer": "ByteLevel",
        "decoder": "ByteLevel",
        "add_prefix_space": False,
        "automatic_bos_eos": False,
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": {
            token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS
        },
        "inspection": inspection.to_dict(),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "tokenizer_sha256": _file_sha256(tokenizer_path),
    }
    write_json(manifest_path, manifest)
    return TokenizerTrainingResult(
        tokenizer_path=tokenizer_path,
        manifest_path=manifest_path,
        requested_vocab_size=vocab_size,
        actual_vocab_size=actual_vocab_size,
        inspection=inspection,
    )


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a saved tokenizer and validate its required invariants."""

    tokenizer_path = Path(path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    validate_tokenizer(tokenizer)
    return tokenizer
