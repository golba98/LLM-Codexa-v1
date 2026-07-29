"""Verification and loading for inference-only Codexa release directories."""

from dataclasses import dataclass
import hmac
import json
from pathlib import Path

from safetensors.torch import load_model
import torch
from tokenizers import Tokenizer

from src.model import LanguageModel, ModelConfig
from src.token_data import file_sha256
from src.tokenizer import BOS_TOKEN, EOS_TOKEN, load_tokenizer


@dataclass(frozen=True)
class ReleaseBundle:
    """Verified model, tokenizer, and metadata loaded for inference."""

    root: Path
    model: LanguageModel
    tokenizer: Tokenizer
    manifest: dict[str, object]


def verify_release_directory(path: str | Path) -> dict[str, object]:
    """Verify release checksums and cross-file metadata consistency."""

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Release directory does not exist: {root}")
    required = {
        "model.safetensors",
        "tokenizer.json",
        "model_config.json",
        "training_state.json",
        "release_manifest.json",
        "SHA256SUMS",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"Release directory is missing: {', '.join(missing)}."
        )

    expected_checksums: dict[str, str] = {}
    for line_number, line in enumerate(
        (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"SHA256SUMS:{line_number}: malformed line.")
        checksum, filename = fields
        filename = filename.lstrip("* ")
        if (
            len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
            or Path(filename).name != filename
            or filename in expected_checksums
        ):
            raise ValueError(f"SHA256SUMS:{line_number}: invalid entry.")
        expected_checksums[filename] = checksum
    required_checked = required - {"SHA256SUMS"}
    if not required_checked <= set(expected_checksums):
        raise ValueError("SHA256SUMS does not cover every required artifact.")
    for filename, expected in expected_checksums.items():
        artifact = root / filename
        if not artifact.is_file():
            raise FileNotFoundError(f"Checksummed artifact is missing: {filename}")
        actual = file_sha256(artifact)
        if not hmac.compare_digest(expected, actual):
            raise ValueError(
                f"Release checksum mismatch for {filename}: "
                f"expected {expected}, got {actual}."
            )

    manifest = json.loads(
        (root / "release_manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(manifest, dict):
        raise ValueError("Release manifest must contain an object.")
    declared = manifest.get("artifacts")
    if not isinstance(declared, dict):
        raise ValueError("Release manifest artifacts must be an object.")
    checksummed_artifacts = set(expected_checksums) - {
        "release_manifest.json"
    }
    if set(declared) != checksummed_artifacts:
        raise ValueError(
            "Release manifest must declare every checksummed artifact exactly "
            "once."
        )
    for filename, checksum in declared.items():
        if (
            not isinstance(filename, str)
            or not isinstance(checksum, str)
            or expected_checksums.get(filename) != checksum
        ):
            raise ValueError(
                "Release manifest checksums disagree with SHA256SUMS."
            )

    tokenizer_checksum = file_sha256(root / "tokenizer.json")
    if manifest.get("tokenizer_sha256") != tokenizer_checksum:
        raise ValueError(
            "Release manifest tokenizer checksum does not match tokenizer.json."
        )
    source_checksum = manifest.get("source_checkpoint_sha256")
    if (
        not isinstance(source_checksum, str)
        or len(source_checksum) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_checksum
        )
    ):
        raise ValueError(
            "Release manifest source checkpoint checksum is invalid."
        )
    source_step = manifest.get("source_checkpoint_optimizer_step")
    if (
        not isinstance(source_step, int)
        or isinstance(source_step, bool)
        or source_step < 0
    ):
        raise ValueError(
            "Release manifest source checkpoint step is invalid."
        )
    training_state = json.loads(
        (root / "training_state.json").read_text(encoding="utf-8")
    )
    if not isinstance(training_state, dict):
        raise ValueError("Release training state must contain an object.")
    if training_state.get("optimizer_step") != source_step:
        raise ValueError(
            "Release training state does not match the manifest checkpoint "
            "step."
        )
    training_stage = manifest.get("training_stage")
    if training_stage not in {"pretraining", "supervised_fine_tuning"}:
        raise ValueError("Release manifest training stage is invalid.")
    if training_stage == "supervised_fine_tuning":
        if (
            not isinstance(manifest.get("chat_template_version"), str)
            or not isinstance(manifest.get("base_checkpoint"), dict)
            or not (root / "CHAT_TEMPLATE.md").is_file()
        ):
            raise ValueError(
                "Instruction release lineage or chat template is incomplete."
            )
    return manifest


def load_release(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> ReleaseBundle:
    """Load a checksum-verified safetensors release for inference."""

    root = Path(path)
    manifest = verify_release_directory(root)
    config_value = json.loads(
        (root / "model_config.json").read_text(encoding="utf-8")
    )
    if not isinstance(config_value, dict):
        raise ValueError("Model configuration must contain an object.")
    try:
        config = ModelConfig(**config_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid release model configuration: {error}") from error

    resolved_device = torch.device(device)
    model = LanguageModel(config).to(resolved_device)
    missing, unexpected = load_model(
        model,
        root / "model.safetensors",
        strict=True,
        device=str(resolved_device),
    )
    if missing or unexpected:
        raise ValueError(
            f"Release weights mismatch: missing={missing}, unexpected={unexpected}."
        )
    tokenizer = load_tokenizer(root / "tokenizer.json")
    if tokenizer.get_vocab_size(with_added_tokens=True) > config.vocab_size:
        raise ValueError("Release tokenizer exceeds the model vocabulary.")
    if tokenizer.token_to_id(BOS_TOKEN) != 1 or tokenizer.token_to_id(EOS_TOKEN) != 2:
        raise ValueError("Release tokenizer must use <bos>=1 and <eos>=2.")
    model.eval()
    return ReleaseBundle(
        root=root,
        model=model,
        tokenizer=tokenizer,
        manifest=manifest,
    )
