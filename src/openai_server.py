"""OpenAI-compatible local inference for native Codexa checkpoints."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import uuid

import torch

from src.checkpointing import load_model_checkpoint, verify_checkpoint_checksum
from src.generate import GenerationConfig, generate_token_ids
from src.model import LanguageModel, ModelConfig
from src.sft import ChatMessage, format_chat_messages
from src.tokenizer import EOS_TOKEN, load_tokenizer
from src.training import resolve_device


MODEL_ID = "codexa-v1-chat"


@dataclass(frozen=True)
class CompletionResult:
    """One generated continuation and its token accounting."""

    text: str
    prompt_tokens: int
    completion_tokens: int


def render_chat_prompt(messages: object) -> str:
    """Render OpenAI messages with the canonical Codexa chat template."""

    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array.")
    parsed: list[ChatMessage] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object.")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(
                f"messages[{index}].role must be system, user, or assistant."
            )
        if not isinstance(content, str) or not content:
            raise ValueError(
                f"messages[{index}].content must be a non-empty string."
            )
        parsed.append(ChatMessage(role=role, content=content.strip()))
    return format_chat_messages(parsed, add_generation_prompt=True)


def _model_config(checkpoint_path: Path) -> ModelConfig:
    verify_checkpoint_checksum(checkpoint_path)
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be an object.")
    config = payload.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Checkpoint model configuration is missing.")
    try:
        return ModelConfig(**config["model"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid checkpoint model configuration: {error}") from error


class CodexaCompletionEngine:
    """Load one native Codexa checkpoint and generate chat continuations."""

    def __init__(
        self,
        checkpoint: str | Path,
        tokenizer: str | Path,
        *,
        device: str = "auto",
        precision: str = "bf16",
    ) -> None:
        checkpoint_path = Path(checkpoint)
        tokenizer_path = Path(tokenizer)
        self.device = resolve_device(device)
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16.")
        if self.device.type == "cpu" and precision != "fp32":
            raise ValueError("CPU serving requires fp32 precision.")
        if (
            precision == "bf16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("CUDA BF16 is not supported by this device.")
        self.precision = precision
        self.config = _model_config(checkpoint_path)
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.model = LanguageModel(self.config).to(
            device=self.device,
            dtype=dtype,
        )
        loaded = load_model_checkpoint(
            checkpoint_path,
            model=self.model,
            map_location=self.device,
        )
        self.tokenizer = load_tokenizer(tokenizer_path)
        if (
            loaded.tokenizer_sha256 is not None
            and loaded.tokenizer_sha256
            != _sha256(tokenizer_path)
        ):
            raise ValueError("Tokenizer checksum does not match the checkpoint.")
        self.eos_token_id = self.tokenizer.token_to_id(EOS_TOKEN)
        if self.eos_token_id != 2:
            raise ValueError("Tokenizer must use <eos>=2.")
        self.model.eval()

    def complete(
        self,
        messages: object,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> CompletionResult:
        """Generate one assistant continuation."""

        prompt = render_chat_prompt(messages)
        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        ).ids
        if not prompt_ids:
            prompt_ids = [1]
        if len(prompt_ids) >= self.config.context_length:
            raise ValueError("Rendered chat prompt exceeds model context length.")
        available = self.config.context_length - len(prompt_ids)
        generated = generate_token_ids(
            self.model,
            torch.tensor(
                [prompt_ids],
                dtype=torch.long,
                device=self.device,
            ),
            eos_token_id=self.eos_token_id,
            config=GenerationConfig(
                max_new_tokens=min(max_tokens, available),
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                repetition_penalty=1.1,
                do_sample=temperature > 0,
                seed=seed,
            ),
        )[0].tolist()
        completion_ids = generated[len(prompt_ids) :]
        return CompletionResult(
            text=self.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            ),
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(completion_ids),
        )


def validate_chat_request(value: object) -> dict[str, object]:
    """Validate the supported OpenAI chat-completions request subset."""

    if not isinstance(value, dict):
        raise ValueError("Request body must be a JSON object.")
    model = value.get("model", MODEL_ID)
    if model != MODEL_ID:
        raise ValueError(f"model must be {MODEL_ID!r}.")
    render_chat_prompt(value.get("messages"))
    max_tokens = value.get("max_tokens", 128)
    temperature = value.get("temperature", 0.8)
    top_p = value.get("top_p", 0.95)
    seed = value.get("seed", 42)
    stream = value.get("stream", False)
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer.")
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or temperature < 0
    ):
        raise ValueError("temperature must be finite and non-negative.")
    if (
        not isinstance(top_p, (int, float))
        or isinstance(top_p, bool)
        or not math.isfinite(float(top_p))
        or not 0 < float(top_p) <= 1
    ):
        raise ValueError("top_p must be in (0, 1].")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not isinstance(stream, bool):
        raise ValueError("stream must be a boolean.")
    return {
        "model": model,
        "messages": value["messages"],
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "seed": seed,
        "stream": stream,
    }


def model_list_response() -> dict[str, object]:
    """Return an OpenAI-compatible model-list payload."""

    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "golba98",
            }
        ],
    }


def chat_completion_response(
    result: CompletionResult,
    *,
    created: int | None = None,
    completion_id: str | None = None,
) -> dict[str, object]:
    """Build a non-streaming OpenAI chat-completions response."""

    return {
        "id": completion_id or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()) if created is None else created,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": (
                result.prompt_tokens + result.completion_tokens
            ),
        },
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
