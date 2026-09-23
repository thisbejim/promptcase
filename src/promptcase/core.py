"""Input, snapshot, and comparison logic for Promptcase."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, BinaryIO

SCHEMA_VERSION = 1
MAX_LINE_BYTES = 10_000_000
MAX_CASES = 100_000
MAX_TOTAL_BYTES = 100_000_000

TOKENIZE_FIELDS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "add_generation_prompt",
        "add_special_tokens",
        "chat_template",
        "chat_template_kwargs",
        "continue_final_message",
        "mm_processor_kwargs",
    }
)
GENERATION_FIELDS = frozenset(
    {
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "metadata",
        "n",
        "presence_penalty",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream",
        "temperature",
        "top_logprobs",
        "top_p",
        "user",
    }
)
SNAPSHOT_FIELDS = frozenset({"schema_version", "id", "request", "result", "fingerprint"})
RESULT_FIELDS = frozenset({"count", "max_model_len", "tokens"})


class PromptcaseError(Exception):
    """A user-actionable input or protocol error."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def parse_json_line(raw: bytes, *, line_number: int, path: str) -> Any:
    if len(raw) > MAX_LINE_BYTES:
        raise PromptcaseError(
            f"{path}:{line_number}: line exceeds the {MAX_LINE_BYTES:,}-byte safety limit"
        )
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PromptcaseError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def read_jsonl(path: str) -> Iterator[tuple[int, Any]]:
    """Read bounded JSONL rows, reporting their original line numbers."""
    try:
        stream: BinaryIO
        if path == "-":
            import sys

            stream = sys.stdin.buffer
        else:
            stream = Path(path).open("rb")
    except OSError as exc:
        raise PromptcaseError(f"cannot open {path}: {exc.strerror or exc}") from exc

    close_stream = path != "-"
    try:
        count = 0
        total_bytes = 0
        line_number = 0
        while True:
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            line_number += 1
            total_bytes += len(raw)
            if total_bytes > MAX_TOTAL_BYTES:
                raise PromptcaseError(
                    f"{path}: file exceeds the {MAX_TOTAL_BYTES:,}-byte safety limit"
                )
            if len(raw) > MAX_LINE_BYTES:
                raise PromptcaseError(
                    f"{path}:{line_number}: line exceeds the {MAX_LINE_BYTES:,}-byte safety limit"
                )
            if not raw.strip():
                continue
            count += 1
            if count > MAX_CASES:
                raise PromptcaseError(f"{path}: more than {MAX_CASES:,} non-empty rows")
            yield line_number, parse_json_line(raw, line_number=line_number, path=path)
    finally:
        if close_stream:
            stream.close()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def normalize_case(value: Any, *, line_number: int, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PromptcaseError(f"{path}:{line_number}: each row must be a JSON object")
    case_id = value.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise PromptcaseError(f"{path}:{line_number}: 'id' must be a non-empty string")

    request = value.get("request")
    if not isinstance(request, dict):
        raise PromptcaseError(f"{path}:{line_number}: 'request' must be a JSON object")
    unknown = set(request) - TOKENIZE_FIELDS - GENERATION_FIELDS
    if unknown:
        keys = ", ".join(sorted(str(key) for key in unknown))
        raise PromptcaseError(
            f"{path}:{line_number}: unsupported request field(s): {keys}. "
            "Promptcase accepts message/template fields and ignores known generation settings."
        )
    if "messages" not in request or not isinstance(request["messages"], list):
        raise PromptcaseError(f"{path}:{line_number}: request.messages must be an array")
    if not request["messages"]:
        raise PromptcaseError(f"{path}:{line_number}: request.messages must not be empty")
    if any(not isinstance(message, dict) for message in request["messages"]):
        raise PromptcaseError(
            f"{path}:{line_number}: every request.messages item must be an object"
        )
    if "model" in request and (
        not isinstance(request["model"], str) or not request["model"].strip()
    ):
        raise PromptcaseError(f"{path}:{line_number}: request.model must be a non-empty string")
    for field in ("add_generation_prompt", "add_special_tokens", "continue_final_message"):
        if field in request and not isinstance(request[field], bool):
            raise PromptcaseError(f"{path}:{line_number}: request.{field} must be boolean")
    if request.get("add_generation_prompt") and request.get("continue_final_message"):
        raise PromptcaseError(
            f"{path}:{line_number}: add_generation_prompt and continue_final_message "
            "cannot both be true"
        )
    for field in ("tools",):
        if field in request and not isinstance(request[field], list):
            raise PromptcaseError(f"{path}:{line_number}: request.{field} must be an array")
    for field in ("chat_template_kwargs", "mm_processor_kwargs"):
        if field in request and not isinstance(request[field], dict):
            raise PromptcaseError(f"{path}:{line_number}: request.{field} must be an object")
    if "chat_template" in request and not isinstance(request["chat_template"], str):
        raise PromptcaseError(f"{path}:{line_number}: request.chat_template must be a string")

    prompt_request = {key: request[key] for key in TOKENIZE_FIELDS if key in request}
    return {"id": case_id, "request": prompt_request}


def validate_result(value: Any, *, case_id: str, path: str, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PromptcaseError(f"{path}:{line_number}: response for {case_id!r} must be an object")
    missing = RESULT_FIELDS - set(value)
    if missing:
        names = ", ".join(sorted(missing))
        raise PromptcaseError(f"{path}:{line_number}: response for {case_id!r} is missing {names}")
    count = value["count"]
    max_model_len = value["max_model_len"]
    tokens = value["tokens"]
    if not _is_int(count) or count < 0:
        raise PromptcaseError(f"{path}:{line_number}: response count for {case_id!r} must be >= 0")
    if not _is_int(max_model_len) or max_model_len <= 0:
        raise PromptcaseError(
            f"{path}:{line_number}: response max_model_len for {case_id!r} must be > 0"
        )
    if not isinstance(tokens, list) or any(not _is_int(token) or token < 0 for token in tokens):
        raise PromptcaseError(
            f"{path}:{line_number}: response tokens for {case_id!r} must be non-negative integers"
        )
    if count != len(tokens):
        raise PromptcaseError(
            f"{path}:{line_number}: response count for {case_id!r} is {count}, "
            f"but tokens contains {len(tokens)} items"
        )
    return {"count": count, "max_model_len": max_model_len, "tokens": tokens}


def make_snapshot(case: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": case["id"],
        "request": case["request"],
        "result": dict(result),
    }
    row["fingerprint"] = snapshot_fingerprint(row)
    return row


def snapshot_fingerprint(row: Mapping[str, Any]) -> str:
    material = {
        "schema_version": row.get("schema_version"),
        "id": row.get("id"),
        "request": row.get("request"),
        "result": row.get("result"),
    }
    encoded = json.dumps(
        material, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_snapshot(value: Any, *, line_number: int, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PromptcaseError(f"{path}:{line_number}: snapshot row must be an object")
    unknown = set(value) - SNAPSHOT_FIELDS
    if unknown:
        keys = ", ".join(sorted(str(key) for key in unknown))
        raise PromptcaseError(f"{path}:{line_number}: unknown snapshot field(s): {keys}")
    if not _is_int(value.get("schema_version")) or value["schema_version"] != SCHEMA_VERSION:
        raise PromptcaseError(
            f"{path}:{line_number}: unsupported schema_version; expected {SCHEMA_VERSION}"
        )
    case = normalize_case(
        {"id": value.get("id"), "request": value.get("request")},
        line_number=line_number,
        path=path,
    )
    result = validate_result(
        value.get("result"), case_id=case["id"], path=path, line_number=line_number
    )
    if isinstance(value.get("result"), dict) and set(value["result"]) - RESULT_FIELDS:
        keys = ", ".join(sorted(set(value["result"]) - RESULT_FIELDS))
        raise PromptcaseError(f"{path}:{line_number}: unknown result field(s): {keys}")
    fingerprint = value.get("fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != snapshot_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "id": case["id"],
            "request": case["request"],
            "result": result,
        }
    ):
        raise PromptcaseError(
            f"{path}:{line_number}: snapshot fingerprint mismatch for {case['id']!r}"
        )
    return make_snapshot(case, result)


def load_cases(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, value in read_jsonl(path):
        case = normalize_case(value, line_number=line_number, path=path)
        if case["id"] in seen:
            raise PromptcaseError(f"{path}:{line_number}: duplicate id {case['id']!r}")
        seen.add(case["id"])
        rows.append(case)
    if not rows:
        raise PromptcaseError(f"{path}: no non-empty JSONL rows")
    return rows


def load_snapshots(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, value in read_jsonl(path):
        row = normalize_snapshot(value, line_number=line_number, path=path)
        if row["id"] in seen:
            raise PromptcaseError(f"{path}:{line_number}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise PromptcaseError(f"{path}: no non-empty JSONL rows")
    return rows


def first_difference(expected: list[int], actual: list[int]) -> dict[str, Any] | None:
    for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
        if left != right:
            return {"index": index, "expected": left, "actual": right}
    if len(expected) != len(actual):
        index = min(len(expected), len(actual))
        return {
            "index": index,
            "expected": expected[index] if index < len(expected) else None,
            "actual": actual[index] if index < len(actual) else None,
        }
    return None


def compare_snapshot_sets(
    expected_rows: Iterable[Mapping[str, Any]], actual_rows: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    expected = {row["id"]: row for row in expected_rows}
    actual = {row["id"]: row for row in actual_rows}
    findings: list[dict[str, Any]] = []
    for case_id in expected:
        if case_id not in actual:
            findings.append({"code": "CASE_MISSING", "id": case_id})
            continue
        old_result = expected[case_id]["result"]
        new_result = actual[case_id]["result"]
        difference = first_difference(old_result["tokens"], new_result["tokens"])
        if difference is not None:
            findings.append(
                {
                    "code": "TOKEN_DRIFT",
                    "id": case_id,
                    "index": difference["index"],
                    "expected_token": difference["expected"],
                    "actual_token": difference["actual"],
                    "expected_count": old_result["count"],
                    "actual_count": new_result["count"],
                }
            )
        if old_result["max_model_len"] != new_result["max_model_len"]:
            findings.append(
                {
                    "code": "MAX_MODEL_LEN_CHANGED",
                    "id": case_id,
                    "expected": old_result["max_model_len"],
                    "actual": new_result["max_model_len"],
                }
            )
    for case_id in actual:
        if case_id not in expected:
            findings.append({"code": "CASE_ADDED", "id": case_id})
    return findings


def encode_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    chunks: list[str] = []
    for row in rows:
        try:
            chunks.append(
                json.dumps(
                    row, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
                )
            )
        except (TypeError, ValueError) as exc:
            raise PromptcaseError(f"cannot serialize JSONL output: {exc}") from exc
    return ("\n".join(chunks) + "\n").encode("utf-8")
