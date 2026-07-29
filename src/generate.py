"""Autoregressive text-generation utilities for decoder-only models."""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class GenerationConfig:
    """Validated decoding settings."""

    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    do_sample: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer.")
        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or not math.isfinite(float(self.temperature))
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be positive and finite.")
        if self.top_k is not None and (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer when supplied.")
        if self.top_p is not None and (
            not isinstance(self.top_p, (int, float))
            or isinstance(self.top_p, bool)
            or not math.isfinite(float(self.top_p))
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be in (0, 1].")
        if (
            not isinstance(self.repetition_penalty, (int, float))
            or isinstance(self.repetition_penalty, bool)
            or not math.isfinite(float(self.repetition_penalty))
            or self.repetition_penalty <= 0
        ):
            raise ValueError("repetition_penalty must be positive and finite.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_token_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply the standard sign-aware repetition penalty once per token ID."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocabulary].")
    if generated_token_ids.ndim != 2:
        raise ValueError("generated_token_ids must have shape [batch, sequence].")
    if logits.shape[0] != generated_token_ids.shape[0]:
        raise ValueError("logits and generated tokens must share a batch size.")
    if penalty <= 0 or not math.isfinite(penalty):
        raise ValueError("penalty must be positive and finite.")
    if penalty == 1.0:
        return logits

    adjusted = logits.clone()
    for batch_index in range(adjusted.shape[0]):
        token_ids = torch.unique(generated_token_ids[batch_index])
        selected = adjusted[batch_index, token_ids]
        adjusted[batch_index, token_ids] = torch.where(
            selected < 0,
            selected * penalty,
            selected / penalty,
        )
    return adjusted


def filter_logits(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Apply top-k and nucleus filtering to one logit row per batch."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocabulary].")
    filtered = logits.clone()
    vocabulary_size = filtered.shape[-1]
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        retained = min(top_k, vocabulary_size)
        threshold = torch.topk(filtered, retained, dim=-1).values[:, -1:]
        filtered.masked_fill_(filtered < threshold, float("-inf"))

    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        sorted_logits, sorted_indices = torch.sort(
            filtered,
            descending=True,
            dim=-1,
        )
        cumulative = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1),
            dim=-1,
        )
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits.masked_fill_(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(1, sorted_indices, sorted_logits)
    return filtered


@torch.inference_mode()
def generate_token_ids(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    eos_token_id: int,
    config: GenerationConfig,
) -> torch.Tensor:
    """Generate continuations for a batch of equally sized prompts."""

    if input_ids.ndim != 2 or input_ids.shape[0] <= 0 or input_ids.shape[1] <= 0:
        raise ValueError("input_ids must have shape [batch, non-empty sequence].")
    if input_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("input_ids must use an integer dtype.")
    model_config = getattr(model, "config", None)
    context_length = getattr(model_config, "context_length", None)
    vocab_size = getattr(model_config, "vocab_size", None)
    if not isinstance(context_length, int) or not isinstance(vocab_size, int):
        raise ValueError("model must expose config.context_length and vocab_size.")
    if input_ids.shape[1] > context_length:
        raise ValueError("Prompt exceeds the model context length.")
    if not 0 <= eos_token_id < vocab_size:
        raise ValueError("eos_token_id is outside the model vocabulary.")
    if config.top_k is not None and config.top_k > vocab_size:
        raise ValueError("top_k cannot exceed the model vocabulary size.")

    device = next(model.parameters()).device
    generated = input_ids.to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    unfinished = torch.ones(
        generated.shape[0],
        dtype=torch.bool,
        device=device,
    )
    was_training = model.training
    try:
        model.eval()
        available = context_length - generated.shape[1]
        for _ in range(min(config.max_new_tokens, available)):
            logits, _loss = model(generated)
            next_logits = logits[:, -1, :].float()
            next_logits = apply_repetition_penalty(
                next_logits,
                generated,
                float(config.repetition_penalty),
            )
            if config.do_sample:
                next_logits = next_logits / float(config.temperature)
                next_logits = filter_logits(
                    next_logits,
                    top_k=config.top_k,
                    top_p=config.top_p,
                )
                probabilities = torch.softmax(next_logits, dim=-1)
                if not torch.isfinite(probabilities).all():
                    raise FloatingPointError(
                        "Sampling probabilities contain NaN or Infinity."
                    )
                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            eos_fill = torch.full_like(next_token, eos_token_id)
            next_token = torch.where(
                unfinished.unsqueeze(1),
                next_token,
                eos_fill,
            )
            generated = torch.cat((generated, next_token), dim=1)
            unfinished &= next_token.squeeze(1) != eos_token_id
            if not unfinished.any():
                break
        return generated
    finally:
        model.train(was_training)
