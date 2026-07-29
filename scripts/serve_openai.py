"""Serve a native Codexa checkpoint through an OpenAI-compatible HTTP API."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.openai_server import (
    CodexaCompletionEngine,
    MODEL_ID,
    chat_completion_response,
    model_list_response,
    validate_chat_request,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1235)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser


def make_handler(engine: CodexaCompletionEngine) -> type[BaseHTTPRequestHandler]:
    """Bind one completion engine to an HTTP request handler."""

    class CodexaRequestHandler(BaseHTTPRequestHandler):
        server_version = "CodexaOpenAI/1.0"

        def _json(self, status: HTTPStatus, value: object) -> None:
            payload = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, status: HTTPStatus, message: str) -> None:
            self._json(
                status,
                {
                    "error": {
                        "message": message,
                        "type": "invalid_request_error",
                    }
                },
            )

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/v1/models":
                self._json(HTTPStatus.OK, model_list_response())
                return
            self._error(HTTPStatus.NOT_FOUND, "Unknown endpoint.")

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._error(HTTPStatus.NOT_FOUND, "Unknown endpoint.")
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if not 0 < content_length <= 1024 * 1024:
                    raise ValueError("Request body size is invalid.")
                request = validate_chat_request(
                    json.loads(self.rfile.read(content_length))
                )
                result = engine.complete(
                    request["messages"],
                    max_tokens=request["max_tokens"],
                    temperature=request["temperature"],
                    top_p=request["top_p"],
                    seed=request["seed"],
                )
                response = chat_completion_response(result)
                if request["stream"]:
                    self.send_response(HTTPStatus.OK)
                    self.send_header(
                        "Content-Type",
                        "text/event-stream; charset=utf-8",
                    )
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    chunk = {
                        "id": response["id"],
                        "object": "chat.completion.chunk",
                        "created": response["created"],
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": result.text,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    self.wfile.write(
                        f"data: {json.dumps(chunk)}\n\n"
                        "data: [DONE]\n\n".encode("utf-8")
                    )
                    self.wfile.flush()
                    return
                self._json(HTTPStatus.OK, response)
            except (
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
            except Exception as error:
                self._error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"Inference failed: {error}",
                )

        def log_message(self, format: str, *arguments: object) -> None:
            print(
                f"{self.client_address[0]} - "
                f"{format % arguments}",
                flush=True,
            )

    return CodexaRequestHandler


def main() -> None:
    arguments = build_argument_parser().parse_args()
    if not 0 < arguments.port <= 65535:
        raise ValueError("--port must be in [1, 65535].")
    engine = CodexaCompletionEngine(
        arguments.checkpoint,
        arguments.tokenizer,
        device=arguments.device,
        precision=arguments.precision,
    )
    server = ThreadingHTTPServer(
        (arguments.host, arguments.port),
        make_handler(engine),
    )
    print(
        f"Serving {MODEL_ID} at "
        f"http://{arguments.host}:{arguments.port}/v1",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Codexa server stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
