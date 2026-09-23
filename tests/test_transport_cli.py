from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from promptcase.cli import main
from promptcase.core import (
    PromptcaseError,
    encode_jsonl,
    load_snapshots,
    make_snapshot,
    normalize_case,
)
from promptcase.transport import tokenize_url


class FakeServer(ThreadingHTTPServer):
    requests: list[dict[str, Any]]
    mode: str


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.server.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
            }
        )
        if self.server.mode == "redirect":
            self.send_response(307)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/capture")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.server.mode == "http-error":
            body = b'{"error":"sensitive request text"}'
            self.send_response(503)
        else:
            if self.server.mode == "drift":
                tokens = [101, 909, 303]
            elif self.server.mode == "bad-count":
                tokens = [101, 202]
            else:
                tokens = [101, 202, 303]
            response_count = len(tokens) + 1 if self.server.mode == "bad-count" else len(tokens)
            body = json.dumps(
                {"count": response_count, "max_model_len": 4096, "tokens": tokens}
            ).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def server():
    instance = FakeServer(("127.0.0.1", 0), Handler)
    instance.requests = []
    instance.mode = "normal"
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def base_url(server: FakeServer) -> str:
    return f"http://127.0.0.1:{server.server_port}"


def input_case() -> dict[str, Any]:
    return {
        "id": "tool-result",
        "request": {
            "model": "fixture-model",
            "messages": [
                {"role": "user", "content": "What is the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "15°C, cloudy"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "add_generation_prompt": True,
            "temperature": 0,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(encode_jsonl(rows))


def test_tokenize_url_handles_v1_base_and_rejects_credentials() -> None:
    assert tokenize_url("http://localhost:8000/v1/") == "http://localhost:8000/tokenize"
    with pytest.raises(PromptcaseError, match="credentials"):
        tokenize_url("http://user:secret@localhost:8000")


def test_record_and_check_use_chat_aware_payload_and_hide_prompt_in_report(
    server: FakeServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = tmp_path / "requests.jsonl"
    snapshots = tmp_path / "snapshots.jsonl"
    write_jsonl(requests, [input_case()])
    monkeypatch.setenv("PROMPTCASE_TEST_KEY", "example-secret")

    assert (
        main(
            [
                "record",
                str(requests),
                "--base-url",
                base_url(server),
                "--output",
                str(snapshots),
                "--api-key-env",
                "PROMPTCASE_TEST_KEY",
            ]
        )
        == 0
    )
    saved = load_snapshots(str(snapshots))
    assert saved[0]["result"]["tokens"] == [101, 202, 303]
    sent = server.requests[0]
    assert sent["path"] == "/tokenize"
    assert sent["body"]["messages"][-1]["content"] == "15°C, cloudy"
    assert sent["body"]["return_token_strs"] is False
    assert sent["headers"]["Authorization"] == "Bearer example-secret"
    assert "example-secret" not in snapshots.read_text(encoding="utf-8")
    assert "temperature" not in sent["body"]

    assert main(["check", str(snapshots), "--base-url", base_url(server), "--format", "json"]) == 0
    report = capsys.readouterr().out
    assert '"ok": true' in report
    assert "15°C, cloudy" not in report


def test_check_fails_with_first_changed_token(server: FakeServer, tmp_path: Path, capsys) -> None:
    row = input_case()
    normalized = normalize_case(row, line_number=1, path="fixture")
    approved = make_snapshot(
        normalized,
        {"count": 3, "max_model_len": 4096, "tokens": [101, 202, 303]},
    )
    path = tmp_path / "snapshots.jsonl"
    write_jsonl(path, [approved])
    server.mode = "drift"

    code = main(["check", str(path), "--base-url", base_url(server)])
    output = capsys.readouterr().out
    assert code == 1
    assert "token 1 changed 202 -> 909" in output
    assert "15°C, cloudy" not in output


def test_check_http_error_does_not_print_response_body(
    server: FakeServer, tmp_path: Path, capsys
) -> None:
    row = input_case()
    normalized = normalize_case(row, line_number=1, path="fixture")
    approved = make_snapshot(
        normalized,
        {"count": 3, "max_model_len": 4096, "tokens": [101, 202, 303]},
    )
    path = tmp_path / "snapshots.jsonl"
    write_jsonl(path, [approved])
    server.mode = "http-error"

    code = main(["check", str(path), "--base-url", base_url(server)])
    captured = capsys.readouterr()
    assert code == 2
    assert "HTTP 503" in captured.out
    assert "sensitive request text" not in captured.out


def test_redirect_is_not_followed_with_prompt_payload(
    server: FakeServer, tmp_path: Path, capsys
) -> None:
    row = normalize_case(input_case(), line_number=1, path="fixture")
    path = tmp_path / "snapshots.jsonl"
    write_jsonl(
        path,
        [make_snapshot(row, {"count": 3, "max_model_len": 4096, "tokens": [101, 202, 303]})],
    )
    server.mode = "redirect"

    code = main(["check", str(path), "--base-url", base_url(server)])
    assert code == 2
    assert "HTTP 307" in capsys.readouterr().out
    assert len(server.requests) == 1


def test_bad_server_count_is_reported_as_protocol_error(
    server: FakeServer, tmp_path: Path, capsys
) -> None:
    row = input_case()
    normalized = normalize_case(row, line_number=1, path="fixture")
    path = tmp_path / "snapshots.jsonl"
    write_jsonl(
        path,
        [make_snapshot(normalized, {"count": 3, "max_model_len": 4096, "tokens": [101, 202, 303]})],
    )
    server.mode = "bad-count"

    code = main(["check", str(path), "--base-url", base_url(server)])
    assert code == 2
    assert "count" in capsys.readouterr().out


def test_max_length_change_is_info_unless_requested(
    server: FakeServer, tmp_path: Path, capsys
) -> None:
    row = normalize_case(input_case(), line_number=1, path="fixture")
    path = tmp_path / "snapshots.jsonl"
    write_jsonl(
        path, [make_snapshot(row, {"count": 3, "max_model_len": 2048, "tokens": [101, 202, 303]})]
    )

    assert main(["check", str(path), "--base-url", base_url(server)]) == 0
    assert "max_model_len 2048 -> 4096" in capsys.readouterr().out
    assert (
        main(["check", str(path), "--base-url", base_url(server), "--fail-on-max-model-len-change"])
        == 1
    )


def test_diff_runs_offline_and_reports_added_missing_and_changed(tmp_path: Path, capsys) -> None:
    row = normalize_case(input_case(), line_number=1, path="fixture")
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    write_jsonl(
        before,
        [
            make_snapshot(row, {"count": 3, "max_model_len": 4096, "tokens": [101, 202, 303]}),
            make_snapshot(
                {"id": "removed", "request": row["request"]},
                {"count": 1, "max_model_len": 4096, "tokens": [5]},
            ),
        ],
    )
    write_jsonl(
        after,
        [
            make_snapshot(row, {"count": 3, "max_model_len": 4096, "tokens": [101, 900, 303]}),
            make_snapshot(
                {"id": "added", "request": row["request"]},
                {"count": 1, "max_model_len": 4096, "tokens": [6]},
            ),
        ],
    )
    assert main(["diff", str(before), str(after)]) == 1
    output = capsys.readouterr().out
    assert "token 1 changed 202 -> 900" in output
    assert "MISSING removed" in output
    assert "ADDED added" in output
