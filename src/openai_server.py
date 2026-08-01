"""OpenAI-compatible local inference for native Codexa chat checkpoints."""

from collections.abc import Callable
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import uuid

import torch

from src.chat_protocol import (
    ASSISTANT_TOKEN,
    CHAT_TEMPLATE_VERSION,
    END_TOKEN,
    SPECIAL_TOKEN_IDS,
    SYSTEM_TOKEN,
    USER_TOKEN,
    ChatMessage,
    chat_special_token_map,
    encode_content_ids,
    encode_chat_messages,
    format_chat_messages,
    validate_chat_messages,
    validate_chat_tokenizer,
)
from src.checkpointing import load_model_checkpoint, verify_checkpoint_checksum
from src.generate import GenerationConfig, generate_sequences
from src.model import LanguageModel, ModelConfig
from src.tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, load_tokenizer
from src.training import resolve_device


MODEL_ID = "golba98/codexa-openai-adapter"
_SUPPORTED_FIELDS = {
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "repetition_penalty",
    "seed",
    "stream",
    "stop",
    "n",
    "logprobs",
    "tools",
    "tool_choice",
}


@dataclass(frozen=True)
class CompletionResult:
    """One generated continuation and its termination/accounting metadata."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    termination_cause: str
    terminating_token_id: int | None
    generated_token_ids: tuple[int, ...] = ()


def parse_chat_messages(messages: object) -> tuple[ChatMessage, ...]:
    """Parse the supported text-only OpenAI message representation."""

    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array.")
    parsed: list[ChatMessage] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object.")
        extra = set(message) - {"role", "content"}
        if extra:
            raise ValueError(
                f"messages[{index}] contains unsupported fields: "
                + ", ".join(sorted(extra))
            )
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(
                f"messages[{index}].role must be system, user, or assistant."
            )
        if not isinstance(content, str):
            raise ValueError(f"messages[{index}].content must be a string.")
        parsed.append(ChatMessage(role=role, content=content))
    validate_chat_messages(parsed, require_assistant_response=False)
    return tuple(parsed)


def render_chat_prompt(messages: object) -> str:
    """Render an OpenAI message array with canonical template version 3.0."""

    return format_chat_messages(
        parse_chat_messages(messages),
        add_generation_prompt=True,
    )


def _checkpoint_header(checkpoint_path: Path) -> tuple[ModelConfig, dict[str, object]]:
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
        model_config = ModelConfig(**config["model"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid checkpoint model configuration: {error}") from error
    return model_config, payload


class _DecodedTextStreamer:
    """Convert stable accumulated token decodes into matching text deltas."""

    def __init__(self, tokenizer, callback: Callable[[str], None]) -> None:
        self.tokenizer = tokenizer
        self.callback = callback
        self.token_ids: list[int] = []
        self.emitted = ""

    def add(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        decoded = self.tokenizer.decode(
            self.token_ids,
            skip_special_tokens=False,
        ).lstrip()
        if "\ufffd" in decoded or not decoded.startswith(self.emitted):
            return
        stable = decoded.rstrip()
        if len(stable) > len(self.emitted):
            delta = stable[len(self.emitted) :]
            self.callback(delta)
            self.emitted = stable

    def finish(self, final_text: str) -> None:
        if final_text.startswith(self.emitted):
            delta = final_text[len(self.emitted) :]
            if delta:
                self.callback(delta)


class CodexaCompletionEngine:
    """Load one compatible SFT checkpoint and generate chat continuations."""

    def __init__(
        self,
        checkpoint: str | Path,
        tokenizer: str | Path,
        *,
        device: str = "auto",
        precision: str = "bf16",
        debug_chat: bool = False,
        allow_legacy_template: bool = False,
    ) -> None:
        checkpoint_path = Path(checkpoint)
        tokenizer_path = Path(tokenizer)
        self.device = resolve_device(device)
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16.")
        if self.device.type == "cpu" and precision != "fp32":
            raise ValueError("CPU serving requires fp32 precision.")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 is not supported by this device.")
        self.precision = precision
        self.debug_chat = debug_chat
        self.config, header = _checkpoint_header(checkpoint_path)
        if header.get("training_stage") != "supervised_fine_tuning":
            raise ValueError(
                "Chat serving requires a supervised-fine-tuning checkpoint."
            )
        template_version = header.get("chat_template_version")
        self.legacy_template = template_version == "2.0" and allow_legacy_template
        if template_version != CHAT_TEMPLATE_VERSION and not self.legacy_template:
            raise ValueError(
                f"Chat serving requires template {CHAT_TEMPLATE_VERSION}."
            )
        if (
            not self.legacy_template
            and header.get("chat_special_token_ids") != chat_special_token_map()
        ):
            raise ValueError("Checkpoint chat special-token metadata is missing or invalid.")
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.model = LanguageModel(self.config).to(device=self.device, dtype=dtype)
        loaded = load_model_checkpoint(
            checkpoint_path,
            model=self.model,
            map_location=self.device,
        )
        self.tokenizer = load_tokenizer(tokenizer_path)
        if not self.legacy_template:
            validate_chat_tokenizer(self.tokenizer)
        if self.config.vocab_size != self.tokenizer.get_vocab_size(True):
            raise ValueError("Model and tokenizer vocabulary sizes do not match.")
        if loaded.tokenizer_sha256 is not None and loaded.tokenizer_sha256 != _sha256(
            tokenizer_path
        ):
            raise ValueError("Tokenizer checksum does not match the checkpoint.")
        self.eos_token_id = self.tokenizer.token_to_id(EOS_TOKEN)
        self.pad_token_id = self.tokenizer.token_to_id(PAD_TOKEN)
        if self.eos_token_id != 2 or self.pad_token_id != 0:
            raise ValueError("Tokenizer must use <pad>=0 and <eos>=2.")
        self.model.eval()

    def complete(
        self,
        messages: object,
        *,
        max_tokens: int,
        temperature: float | None,
        top_p: float | None,
        repetition_penalty: float,
        seed: int,
        no_repeat_ngram_size: int | None = 3,
        stop: tuple[str, ...] = (),
        on_text_delta: Callable[[str], None] | None = None,
    ) -> CompletionResult:
        """Generate one assistant continuation with shared stop semantics."""

        parsed = parse_chat_messages(messages)
        if self.legacy_template:
            prompt = "\n".join(
                [f"{message.role.title()}: {message.content.strip()}" for message in parsed]
                + ["Assistant:"]
            )
            prompt_ids = self.tokenizer.encode(
                prompt, add_special_tokens=False
            ).ids
        else:
            prompt = format_chat_messages(parsed, add_generation_prompt=True)
            prompt_ids, _ = encode_chat_messages(
                parsed,
                tokenizer=self.tokenizer,
                add_generation_prompt=True,
            )
        available = self.config.context_length - len(prompt_ids)
        if available <= 0:
            raise ValueError("Rendered chat prompt exceeds model context length.")
        if max_tokens > available:
            raise ValueError(
                f"max_tokens exceeds remaining context capacity ({available})."
            )

        if self.legacy_template:
            stop_sequences = [
                tuple(self.tokenizer.encode(marker, add_special_tokens=False).ids)
                for marker in ("\nSystem:", "\nUser:", "\nAssistant:")
            ]
        else:
            stop_sequences = [
                (SPECIAL_TOKEN_IDS[PAD_TOKEN],),
                (SPECIAL_TOKEN_IDS[BOS_TOKEN],),
                (SPECIAL_TOKEN_IDS[UNK_TOKEN],),
                (SPECIAL_TOKEN_IDS[END_TOKEN],),
                (SPECIAL_TOKEN_IDS[SYSTEM_TOKEN],),
                (SPECIAL_TOKEN_IDS[USER_TOKEN],),
                (SPECIAL_TOKEN_IDS[ASSISTANT_TOKEN],),
            ]
        fallback_stops: list[str] = []
        for value in stop:
            token_ids = tuple(encode_content_ids(self.tokenizer, value))
            if token_ids and self.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
            ) == value:
                stop_sequences.append(token_ids)
            else:
                fallback_stops.append(value)

        streamer = (
            _DecodedTextStreamer(self.tokenizer, on_text_delta)
            if on_text_delta is not None and not fallback_stops
            else None
        )
        result = generate_sequences(
            self.model,
            torch.tensor([prompt_ids], dtype=torch.long, device=self.device),
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            stop_sequences=stop_sequences,
            config=GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=temperature is not None,
            seed=seed,
        ),
            token_callback=None if streamer is None else streamer.add,
        ).sequences[0]
        text = self.tokenizer.decode(
            list(result.visible_token_ids),
            skip_special_tokens=False,
        ).strip()
        finish_reason = result.finish_reason
        termination_cause = result.termination_cause
        for fallback in fallback_stops:
            position = text.find(fallback)
            if position >= 0:
                text = text[:position].rstrip()
                finish_reason = "stop"
                termination_cause = "client_stop"
                break
        if streamer is not None:
            streamer.finish(text)
        elif on_text_delta is not None and text:
            on_text_delta(text)

        terminating_token_id = result.terminating_token_id
        if terminating_token_id == self.eos_token_id:
            termination_cause = "eos"
        elif not self.legacy_template and terminating_token_id == SPECIAL_TOKEN_IDS[END_TOKEN]:
            termination_cause = "end"
        elif not self.legacy_template and terminating_token_id in {
            SPECIAL_TOKEN_IDS[SYSTEM_TOKEN],
            SPECIAL_TOKEN_IDS[USER_TOKEN],
            SPECIAL_TOKEN_IDS[ASSISTANT_TOKEN],
        }:
            termination_cause = "role_token"
        diagnostics: dict[str, object] = {
            "input_token_count": len(prompt_ids),
            "newly_generated_token_count": result.generated_token_count,
            "special_token_ids": (
                {PAD_TOKEN: 0, BOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
                if self.legacy_template
                else chat_special_token_map()
            ),
            "chat_template_version": "2.0" if self.legacy_template else CHAT_TEMPLATE_VERSION,
            "terminating_token_id": terminating_token_id,
            "terminating_token": (
                None
                if terminating_token_id is None
                else self.tokenizer.id_to_token(terminating_token_id)
            ),
            "termination_cause": termination_cause,
            "finish_reason": finish_reason,
        }
        if self.debug_chat:
            diagnostics["rendered_chat_template"] = prompt
        print(json.dumps({"chat_generation": diagnostics}), flush=True)
        return CompletionResult(
            text=text,
            prompt_tokens=len(prompt_ids),
            completion_tokens=result.generated_token_count,
            finish_reason=finish_reason,
            termination_cause=termination_cause,
            terminating_token_id=terminating_token_id,
            generated_token_ids=result.token_ids,
        )


def _number(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < minimum
        or (maximum is not None and float(value) > maximum)
    ):
        range_text = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be finite and in {range_text}.")
    return float(value)


def validate_chat_request(value: object) -> dict[str, object]:
    """Validate the supported OpenAI chat-completions request subset."""

    if not isinstance(value, dict):
        raise ValueError("Request body must be a JSON object.")
    unknown = set(value) - _SUPPORTED_FIELDS
    if unknown:
        raise ValueError("Unsupported request fields: " + ", ".join(sorted(unknown)))
    model = value.get("model", MODEL_ID)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string.")
    parse_chat_messages(value.get("messages"))
    if "max_tokens" in value and "max_completion_tokens" in value:
        raise ValueError("Use max_tokens or max_completion_tokens, not both.")
    max_tokens = value.get("max_completion_tokens", value.get("max_tokens", 128))
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer.")
    raw_temperature = value.get("temperature")
    temperature: float | None
    if raw_temperature is None or raw_temperature == 0:
        temperature = None
    else:
        temperature = _number(raw_temperature, name="temperature", minimum=0.0, maximum=2.0)
        if temperature == 0:
            temperature = None
    raw_top_p = value.get("top_p")
    if temperature is None:
        if raw_top_p not in {None, 1, 1.0}:
            raise ValueError("top_p below 1 requires positive temperature sampling.")
        top_p = None
    else:
        top_p = 0.9 if raw_top_p is None else _number(
            raw_top_p,
            name="top_p",
            minimum=1e-12,
            maximum=1.0,
        )
    default_penalty = 1.0 if temperature is None else 1.075
    repetition_penalty = _number(
        value.get("repetition_penalty", default_penalty),
        name="repetition_penalty",
        minimum=1e-12,
    )
    seed = value.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    stream = value.get("stream", False)
    if not isinstance(stream, bool):
        raise ValueError("stream must be a boolean.")
    stop_value = value.get("stop")
    if stop_value is None:
        stop: tuple[str, ...] = ()
    elif isinstance(stop_value, str):
        stop = (stop_value,)
    elif isinstance(stop_value, list) and all(isinstance(item, str) for item in stop_value):
        stop = tuple(stop_value)
    else:
        raise ValueError("stop must be a string or an array of strings.")
    if len(stop) > 4 or any(not item for item in stop):
        raise ValueError("stop must contain one to four non-empty strings.")
    if value.get("n", 1) != 1:
        raise ValueError("Only n=1 is supported.")
    if value.get("logprobs") not in {None, False}:
        raise ValueError("logprobs are not supported.")
    tools = value.get("tools")
    if tools is not None and tools != [] and tools != ():
        raise ValueError("tools are not supported.")
    if value.get("tool_choice") is not None:
        raise ValueError("tool_choice is not supported.")
    return {
        "model": model,
        "messages": value["messages"],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "seed": seed,
        "stream": stream,
        "stop": stop,
    }


def model_list_response() -> dict[str, object]:
    """Return an OpenAI-compatible model-list payload."""

    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "golba98"}],
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
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "finish_reason": result.finish_reason,
        }],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
