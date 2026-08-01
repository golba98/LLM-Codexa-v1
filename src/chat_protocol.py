"""Canonical token-level chat protocol shared by training and inference."""

from dataclasses import dataclass
from typing import Sequence

from tokenizers import AddedToken, Tokenizer

from src.tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN


CHAT_TEMPLATE_VERSION = "3.0"
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"
CHAT_SPECIAL_TOKENS = (
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    END_TOKEN,
)
SPECIAL_TOKEN_IDS = {
    PAD_TOKEN: 0,
    BOS_TOKEN: 1,
    EOS_TOKEN: 2,
    UNK_TOKEN: 3,
    SYSTEM_TOKEN: 8192,
    USER_TOKEN: 8193,
    ASSISTANT_TOKEN: 8194,
    END_TOKEN: 8195,
}
CHAT_VOCAB_SIZE = 8196
CHAT_ROLES = ("system", "user", "assistant")
ROLE_TOKENS = {
    "system": SYSTEM_TOKEN,
    "user": USER_TOKEN,
    "assistant": ASSISTANT_TOKEN,
}


@dataclass(frozen=True)
class ChatMessage:
    """One validated text message in a Codexa conversation."""

    role: str
    content: str


def normalize_content(content: str) -> str:
    """Normalize line endings without changing meaningful whitespace."""

    return content.replace("\r\n", "\n").replace("\r", "\n")


def validate_chat_messages(
    messages: Sequence[ChatMessage],
    *,
    require_assistant_response: bool,
) -> None:
    """Validate the optional-system then alternating user/assistant schema."""

    if not messages:
        raise ValueError("Chat messages must not be empty.")
    expected = "user"
    for index, message in enumerate(messages):
        if message.role not in CHAT_ROLES:
            raise ValueError(
                f"messages[{index}].role must be system, user, or assistant."
            )
        if not isinstance(message.content, str):
            raise ValueError(f"messages[{index}].content must be a string.")
        if message.role == "system":
            if index != 0:
                raise ValueError(
                    "A system message is allowed only as the first message."
                )
            continue
        if not message.content.strip():
            raise ValueError(
                f"messages[{index}].content must be a non-empty string."
            )
        if message.role != expected:
            raise ValueError(
                f"messages[{index}].role must be {expected}; "
                f"received {message.role}."
            )
        expected = "assistant" if message.role == "user" else "user"
    required_last_role = "assistant" if require_assistant_response else "user"
    if messages[-1].role != required_last_role:
        description = "training conversation" if require_assistant_response else "generation prompt"
        raise ValueError(f"A {description} must end with {required_last_role}.")


def _with_system_message(messages: Sequence[ChatMessage]) -> tuple[ChatMessage, ...]:
    normalized = tuple(
        ChatMessage(message.role, normalize_content(message.content))
        for message in messages
    )
    if normalized[0].role == "system":
        return normalized
    return (ChatMessage("system", ""), *normalized)


def format_chat_messages(
    messages: Sequence[ChatMessage],
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Render the exact human-readable form of chat template version 3.0."""

    validate_chat_messages(
        messages,
        require_assistant_response=not add_generation_prompt,
    )
    normalized = _with_system_message(messages)
    pieces = [BOS_TOKEN]
    for message in normalized:
        pieces.extend(
            (
                ROLE_TOKENS[message.role],
                "\n",
                message.content,
                END_TOKEN,
                "\n",
            )
        )
    if add_generation_prompt:
        pieces.extend((ASSISTANT_TOKEN, "\n"))
    return "".join(pieces)


def validate_chat_tokenizer(tokenizer: Tokenizer) -> None:
    """Require the canonical vocabulary size and every deterministic ID."""

    for token, expected_id in SPECIAL_TOKEN_IDS.items():
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            raise ValueError(
                f"{token} must have token ID {expected_id}; got {actual_id!r}."
            )
    actual_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_size != CHAT_VOCAB_SIZE:
        raise ValueError(
            f"Chat tokenizer vocabulary must contain {CHAT_VOCAB_SIZE} tokens; "
            f"got {actual_size}."
        )


def extend_tokenizer_for_chat(tokenizer: Tokenizer) -> Tokenizer:
    """Append the four canonical role tokens to an 8,192-token tokenizer."""

    for token, expected_id in tuple(SPECIAL_TOKEN_IDS.items())[:4]:
        if tokenizer.token_to_id(token) != expected_id:
            raise ValueError(f"Base tokenizer must preserve {token}={expected_id}.")
    size = tokenizer.get_vocab_size(with_added_tokens=True)
    if size == CHAT_VOCAB_SIZE:
        validate_chat_tokenizer(tokenizer)
        return tokenizer
    if size != 8192:
        raise ValueError(
            "Chat tokens can only be appended to an exact 8,192-token base "
            f"vocabulary; got {size}."
        )
    added = tokenizer.add_special_tokens(
        [
            AddedToken(token, special=True, normalized=False)
            for token in CHAT_SPECIAL_TOKENS
        ]
    )
    if added != len(CHAT_SPECIAL_TOKENS):
        raise ValueError("Tokenizer already contains a conflicting chat token.")
    validate_chat_tokenizer(tokenizer)
    return tokenizer


def _encode_content(tokenizer: Tokenizer, content: str) -> list[int]:
    # In tokenizers, encode_special_tokens=True means special-looking text in
    # user content is encoded lexically instead of being recognized as control.
    return tokenizer.encode(
        content,
        add_special_tokens=False,
    ).ids


def content_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    """Clone a tokenizer so role-looking content remains ordinary text."""

    clone = Tokenizer.from_str(tokenizer.to_str())
    clone.encode_special_tokens = True
    return clone


def encode_content_ids(tokenizer: Tokenizer, content: str) -> list[int]:
    """Encode untrusted message text without recognizing control tokens."""

    return _encode_content(content_tokenizer(tokenizer), content)


def encode_chat_messages(
    messages: Sequence[ChatMessage],
    *,
    tokenizer: Tokenizer,
    add_generation_prompt: bool,
    assistant_only_labels: bool = False,
    ignore_index: int = -100,
) -> tuple[list[int], list[int]]:
    """Encode chat structure manually, optionally producing SFT labels."""

    validate_chat_tokenizer(tokenizer)
    validate_chat_messages(
        messages,
        require_assistant_response=not add_generation_prompt,
    )
    normalized = _with_system_message(messages)
    bos_id = SPECIAL_TOKEN_IDS[BOS_TOKEN]
    end_id = SPECIAL_TOKEN_IDS[END_TOKEN]
    input_ids = [bos_id]
    labels = [ignore_index]

    def append_masked(values: Sequence[int]) -> None:
        input_ids.extend(values)
        labels.extend([ignore_index] * len(values))

    lexical_tokenizer = content_tokenizer(tokenizer)
    newline_ids = _encode_content(lexical_tokenizer, "\n")
    for message in normalized:
        append_masked([SPECIAL_TOKEN_IDS[ROLE_TOKENS[message.role]]])
        append_masked(newline_ids)
        content_ids = _encode_content(lexical_tokenizer, message.content)
        input_ids.extend(content_ids)
        if assistant_only_labels and message.role == "assistant":
            labels.extend(content_ids)
            input_ids.append(end_id)
            labels.append(end_id)
        else:
            labels.extend([ignore_index] * len(content_ids))
            append_masked([end_id])
        append_masked(newline_ids)
    if add_generation_prompt:
        append_masked([SPECIAL_TOKEN_IDS[ASSISTANT_TOKEN]])
        append_masked(newline_ids)
    return input_ids, labels


def chat_special_token_map() -> dict[str, int]:
    """Return a copy suitable for manifests and debug diagnostics."""

    return dict(SPECIAL_TOKEN_IDS)
