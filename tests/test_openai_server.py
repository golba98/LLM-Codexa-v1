"""Tests for Codexa's OpenAI-compatible response layer."""

import json

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


def main() -> None:
    messages = [
        {"role": "system", "content": "Write a short story."},
        {"role": "user", "content": "A little fox found a key."},
    ]
    assert render_chat_prompt(messages) == (
        "System: Write a short story.\n"
        "User: A little fox found a key.\n"
        "Assistant:"
    )
    assert render_chat_prompt(
        [
            {"role": "user", "content": "Remember blue."},
            {"role": "assistant", "content": "I will remember blue."},
            {"role": "user", "content": "What color did I name?"},
        ]
    ) == (
        "User: Remember blue.\n"
        "Assistant: I will remember blue.\n"
        "User: What color did I name?\n"
        "Assistant:"
    )
    request = validate_chat_request(
        {
            "model": MODEL_ID,
            "messages": messages,
            "max_tokens": 32,
            "temperature": 0.8,
            "top_p": 0.9,
            "seed": 7,
            "stream": True,
        }
    )
    assert request["max_tokens"] == 32
    assert request["stream"] is True
    _raises(
        ValueError,
        lambda: validate_chat_request(
            {"model": "wrong", "messages": messages}
        ),
    )
    _raises(
        ValueError,
        lambda: render_chat_prompt(
            [{"role": "tool", "content": "unsupported"}]
        ),
    )
    _raises(
        ValueError,
        lambda: validate_chat_request(
            {"messages": messages, "temperature": -1}
        ),
    )

    models = model_list_response()
    assert models["data"][0]["id"] == MODEL_ID
    response = chat_completion_response(
        CompletionResult(
            text="The fox opened a tiny door.",
            prompt_tokens=12,
            completion_tokens=7,
        ),
        created=123,
        completion_id="chatcmpl-test",
    )
    assert response["id"] == "chatcmpl-test"
    assert response["choices"][0]["message"]["content"].startswith("The fox")
    assert response["usage"]["total_tokens"] == 19
    assert json.loads(json.dumps(response)) == response

    print("All OpenAI-compatible server tests passed.")


if __name__ == "__main__":
    main()
