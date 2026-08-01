"""Regression tests for Codexa's OpenAI-compatible chat routes."""

from http.server import ThreadingHTTPServer
import json
import re
from threading import Thread
from urllib.request import Request, urlopen

from scripts.serve_openai import make_handler
from src.chat_protocol import CHAT_TEMPLATE_VERSION, END_TOKEN, SPECIAL_TOKEN_IDS
from src.openai_server import (
    CompletionResult,
    MODEL_ID,
    chat_completion_response,
    model_list_response,
    render_chat_prompt,
    validate_chat_request,
)


def _raises(exception_type: type[BaseException], operation) -> None:
    try:
        operation()
    except exception_type:
        return
    raise AssertionError(f"Expected {exception_type.__name__}.")


class ScriptedEngine:
    """Exercise HTTP mechanics without prompt-specific production code."""

    def complete(self, messages, *, max_tokens, on_text_delta=None, **_settings):
        latest = next(
            message["content"]
            for message in reversed(messages)
            if message["role"] == "user"
        )
        if max_tokens == 1:
            text = "A"
            finish_reason = "length"
            cause = "length"
            terminator = None
        elif latest == "hi":
            text = "Hi!"
            finish_reason = "stop"
            cause = "end"
            terminator = SPECIAL_TOKEN_IDS[END_TOKEN]
        elif "exactly one word" in latest:
            text = "hello"
            finish_reason = "stop"
            cause = "end"
            terminator = SPECIAL_TOKEN_IDS[END_TOKEN]
        elif "2 + 2" in latest:
            text = "4"
            finish_reason = "stop"
            cause = "end"
            terminator = SPECIAL_TOKEN_IDS[END_TOKEN]
        elif "name did I give" in latest:
            text = "Sam."
            finish_reason = "stop"
            cause = "end"
            terminator = SPECIAL_TOKEN_IDS[END_TOKEN]
        elif "<|assistant|>" in latest:
            text = "Yes."
            finish_reason = "stop"
            cause = "end"
            terminator = SPECIAL_TOKEN_IDS[END_TOKEN]
        elif latest == "Just say hi":
            text = "Hi."
            finish_reason = "stop"
            cause = "end"
            terminator = SPECIAL_TOKEN_IDS[END_TOKEN]
        else:
            text = "Light scatters in the atmosphere, making the sky look blue."
            finish_reason = "stop"
            cause = "end"
            terminator = SPECIAL_TOKEN_IDS[END_TOKEN]
        if on_text_delta is not None:
            midpoint = max(1, len(text) // 2)
            on_text_delta(text[:midpoint])
            on_text_delta(text[midpoint:])
        return CompletionResult(
            text=text,
            prompt_tokens=12,
            completion_tokens=max(1, len(text.split())),
            finish_reason=finish_reason,
            termination_cause=cause,
            terminating_token_id=terminator,
        )


def _post(port: int, body: dict[str, object]) -> tuple[dict[str, object], str]:
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        text = response.read().decode("utf-8")
    if body.get("stream"):
        events = [
            line.removeprefix("data: ")
            for line in text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        chunks = [json.loads(event) for event in events]
        content = "".join(
            chunk["choices"][0]["delta"].get("content", "")
            for chunk in chunks
        )
        terminal = chunks[-1]
        return terminal, content
    payload = json.loads(text)
    return payload, payload["choices"][0]["message"]["content"]


def test_request_protocol_and_response_schema() -> None:
    messages = [
        {"role": "system", "content": "Write briefly."},
        {"role": "user", "content": "Hello"},
    ]
    prompt = render_chat_prompt(messages)
    assert prompt == (
        "<bos><|system|>\nWrite briefly.<|end|>\n"
        "<|user|>\nHello<|end|>\n<|assistant|>\n"
    )
    assert CHAT_TEMPLATE_VERSION == "3.0"
    literal = render_chat_prompt(
        [{"role": "user", "content": "Is <|assistant|> text?"}]
    )
    assert literal.count("<|assistant|>") == 2

    deterministic = validate_chat_request({"messages": messages})
    assert deterministic["temperature"] is None
    assert deterministic["top_p"] is None
    assert deterministic["max_tokens"] == 128
    assert deterministic["seed"] == 42
    sampled = validate_chat_request(
        {"messages": messages, "temperature": 0.6}
    )
    assert sampled["top_p"] == 0.9
    assert sampled["repetition_penalty"] == 1.075
    alias = validate_chat_request(
        {"messages": messages, "max_completion_tokens": 9}
    )
    assert alias["max_tokens"] == 9
    assert validate_chat_request(
        {"model": "client-alias", "messages": messages}
    )["model"] == "client-alias"
    _raises(
        ValueError,
        lambda: validate_chat_request(
            {"messages": messages, "max_tokens": 3, "max_completion_tokens": 3}
        ),
    )
    _raises(
        ValueError,
        lambda: validate_chat_request(
            {"messages": messages, "temperature": 0, "top_p": 0.9}
        ),
    )
    _raises(
        ValueError,
        lambda: validate_chat_request({"messages": messages, "stop": ["a"] * 5}),
    )
    _raises(
        ValueError,
        lambda: validate_chat_request({"messages": messages, "tools": [{}]}),
    )

    result = CompletionResult(
        text="Hi!",
        prompt_tokens=10,
        completion_tokens=2,
        finish_reason="stop",
        termination_cause="end",
        terminating_token_id=8195,
    )
    response = chat_completion_response(
        result,
        created=123,
        completion_id="chatcmpl-test",
    )
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"]["total_tokens"] == 12
    assert model_list_response()["data"][0]["id"] == MODEL_ID


def test_streaming_and_non_streaming_regressions() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ScriptedEngine()))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    cases = [
        ([{"role": "user", "content": "hi"}], "Hi!", "stop"),
        ([{"role": "user", "content": "Just say hi"}], "Hi.", "stop"),
        ([{"role": "user", "content": "Answer with exactly one word: hello"}], "hello", "stop"),
        ([{"role": "user", "content": "What is 2 + 2?"}], "4", "stop"),
        ([{"role": "user", "content": "Why is the sky blue?"}], "Light scatters", "stop"),
        ([
            {"role": "user", "content": "My name is Sam."},
            {"role": "assistant", "content": "Nice to meet you, Sam."},
            {"role": "user", "content": "What name did I give you?"},
        ], "Sam.", "stop"),
        ([{"role": "user", "content": "Is <|assistant|> literal text?"}], "Yes.", "stop"),
    ]
    try:
        for stream in (False, True):
            for messages, expected, finish_reason in cases:
                terminal, text = _post(
                    port,
                    {"model": MODEL_ID, "messages": messages, "stream": stream},
                )
                assert text.startswith(expected)
                actual_reason = terminal["choices"][0]["finish_reason"]
                assert actual_reason == finish_reason
            terminal, text = _post(
                port,
                {
                    "messages": [{"role": "user", "content": "Continue"}],
                    "max_tokens": 1,
                    "stream": stream,
                },
            )
            assert text == "A"
            assert terminal["choices"][0]["finish_reason"] == "length"

        for _ in range(20):
            payload, text = _post(
                port,
                {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert text == "Hi!"
            assert payload["choices"][0]["finish_reason"] == "stop"
            assert "hi\n" not in text.lower()
            assert not re.search(r"(?i)\b(user|assistant|system)\s*:", text)
            assert not re.search(r"[!?.,]{4,}", text)
            assert all(
                fragment not in text
                for fragment in ("Navelia", "Novalgin", "<|", "product")
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main() -> None:
    test_request_protocol_and_response_schema()
    test_streaming_and_non_streaming_regressions()
    print("All OpenAI-compatible server tests passed.")


if __name__ == "__main__":
    main()
