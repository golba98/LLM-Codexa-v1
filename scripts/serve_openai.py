"""Serve a native Codexa checkpoint through an OpenAI-compatible HTTP API."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import time
import uuid


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
    parser.add_argument(
        "--debug-chat",
        action="store_true",
        help="Log rendered prompts; may expose private user content.",
    )
    parser.add_argument(
        "--allow-legacy-template",
        action="store_true",
        help="Temporarily serve a template-2.0 SFT checkpoint.",
    )
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

        def _sse(self, value: object) -> None:
            self.wfile.write(
                f"data: {json.dumps(value, ensure_ascii=False)}\n\n".encode(
                    "utf-8"
                )
            )
            self.wfile.flush()

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
                if request["stream"]:
                    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
                    created = int(time.time())
                    self.send_response(HTTPStatus.OK)
                    self.send_header(
                        "Content-Type",
                        "text/event-stream; charset=utf-8",
                    )
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    def chunk(delta: dict[str, str], finish_reason=None):
                        return {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [{
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish_reason,
                            }],
                        }

                    self._sse(chunk({"role": "assistant"}))

                    def send_delta(text: str) -> None:
                        if text:
                            self._sse(chunk({"content": text}))

                    result = engine.complete(
                        request["messages"],
                        max_tokens=request["max_tokens"],
                        temperature=request["temperature"],
                        top_p=request["top_p"],
                        repetition_penalty=request["repetition_penalty"],
                        seed=request["seed"],
                        stop=request["stop"],
                        on_text_delta=send_delta,
                    )
                    self._sse(chunk({}, result.finish_reason))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    self.close_connection = True
                    return
                result = engine.complete(
                    request["messages"],
                    max_tokens=request["max_tokens"],
                    temperature=request["temperature"],
                    top_p=request["top_p"],
                    repetition_penalty=request["repetition_penalty"],
                    seed=request["seed"],
                    stop=request["stop"],
                )
                response = chat_completion_response(result)
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
        debug_chat=arguments.debug_chat,
        allow_legacy_template=arguments.allow_legacy_template,
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
