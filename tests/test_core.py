from __future__ import annotations

import json

import pytest

from promptcase.core import (
    PromptcaseError,
    compare_snapshot_sets,
    encode_jsonl,
    first_difference,
    load_cases,
    load_snapshots,
    make_snapshot,
    normalize_case,
    normalize_snapshot,
)


def case(case_id: str = "c1", text: str = "hello") -> dict[str, object]:
    return {
        "id": case_id,
        "request": {
            "model": "fixture-model",
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
        },
    }


def snapshot(case_id: str = "c1", tokens: list[int] | None = None, max_len: int = 2048):
    normalized = normalize_case(case(), line_number=1, path="fixture")
    return make_snapshot(
        {"id": case_id, "request": normalized["request"]},
        {"count": len(tokens or [1, 2]), "max_model_len": max_len, "tokens": tokens or [1, 2]},
    )


def test_normalize_case_ignores_generation_settings() -> None:
    normalized = normalize_case(case(), line_number=1, path="requests.jsonl")
    assert normalized["request"] == {
        "model": "fixture-model",
        "messages": [{"role": "user", "content": "hello"}],
    }


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"id": "", "request": {"messages": [{"role": "user", "content": "x"}]}}, "non-empty"),
        ({"id": "x", "request": {"messages": []}}, "must not be empty"),
        (
            {
                "id": "x",
                "request": {"messages": [{"role": "user", "content": "x"}], "mystery": 1},
            },
            "unsupported request field",
        ),
        (
            {
                "id": "x",
                "request": {
                    "messages": [{"role": "user", "content": "x"}],
                    "tool_choice": "required",
                },
            },
            "unsupported request field",
        ),
        (
            {
                "id": "x",
                "request": {
                    "messages": [{"role": "user", "content": "x"}],
                    "add_generation_prompt": True,
                    "continue_final_message": True,
                },
            },
            "cannot both be true",
        ),
    ],
)
def test_normalize_case_rejects_invalid_shapes(payload, message: str) -> None:
    with pytest.raises(PromptcaseError, match=message):
        normalize_case(payload, line_number=4, path="requests.jsonl")


def test_load_cases_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_bytes(encode_jsonl([case(), case()]))
    with pytest.raises(PromptcaseError, match="duplicate id"):
        load_cases(str(path))


def test_jsonl_rejects_duplicate_object_keys(tmp_path) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_text(
        '{"id":"a","id":"b","request":{"messages":[{"role":"user","content":"x"}]}}\n',
        encoding="utf-8",
    )
    with pytest.raises(PromptcaseError, match="duplicate object key"):
        load_cases(str(path))


def test_snapshot_fingerprint_detects_edits(tmp_path) -> None:
    row = snapshot()
    row["result"]["tokens"][0] = 99
    path = tmp_path / "snapshots.jsonl"
    path.write_bytes(encode_jsonl([row]))
    with pytest.raises(PromptcaseError, match="fingerprint mismatch"):
        load_snapshots(str(path))


def test_snapshot_rejects_boolean_token_ids() -> None:
    value = snapshot(tokens=[1, 2])
    value["result"]["tokens"] = [1, True]
    from promptcase.core import snapshot_fingerprint

    value["fingerprint"] = snapshot_fingerprint(value)
    with pytest.raises(PromptcaseError, match="non-negative integers"):
        normalize_snapshot(value, line_number=1, path="snapshot.jsonl")


@pytest.mark.parametrize(
    "expected, actual, wanted",
    [
        ([1, 2, 3], [1, 9, 3], {"index": 1, "expected": 2, "actual": 9}),
        ([1], [1, 2], {"index": 1, "expected": None, "actual": 2}),
        ([1, 2], [1], {"index": 1, "expected": 2, "actual": None}),
        ([1], [1], None),
    ],
)
def test_first_difference(expected, actual, wanted) -> None:
    assert first_difference(expected, actual) == wanted


def test_compare_snapshots_reports_drift_and_case_changes() -> None:
    findings = compare_snapshot_sets(
        [snapshot("same", [1, 2]), snapshot("gone", [3])],
        [snapshot("same", [1, 7]), snapshot("new", [4])],
    )
    assert [item["code"] for item in findings] == ["TOKEN_DRIFT", "CASE_MISSING", "CASE_ADDED"]
    assert findings[0]["index"] == 1


def test_metadata_change_is_reported_separately() -> None:
    findings = compare_snapshot_sets([snapshot(max_len=2048)], [snapshot(max_len=4096)])
    assert findings == [
        {"code": "MAX_MODEL_LEN_CHANGED", "id": "c1", "expected": 2048, "actual": 4096}
    ]


def test_encode_jsonl_is_stable_and_unicode_safe() -> None:
    row = snapshot()
    encoded = encode_jsonl([row])
    assert encoded.endswith(b"\n")
    assert b"\\u" not in encoded
    assert json.loads(encoded) == row
