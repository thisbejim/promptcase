"""Command-line interface for recording and checking prompt snapshots."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from promptcase import __version__
from promptcase.core import (
    PromptcaseError,
    compare_snapshot_sets,
    encode_jsonl,
    load_cases,
    load_snapshots,
    make_snapshot,
    validate_result,
)
from promptcase.transport import post_tokenize


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0 or parsed == float("inf") or parsed != parsed:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptcase",
        description="Snapshot and regression-test the token IDs produced by a vLLM chat template.",
    )
    parser.add_argument("--version", action="version", version=f"promptcase {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser("record", help="record chat requests as token-ID snapshots")
    record.add_argument("requests", help="JSONL file of {id, request} cases, or - for stdin")
    record.add_argument(
        "--base-url", required=True, help="vLLM server base URL (without /tokenize)"
    )
    record.add_argument("--output", default="-", help="snapshot JSONL path (default: stdout)")
    add_http_options(record)

    check = commands.add_parser("check", help="replay snapshots and fail on token-ID drift")
    check.add_argument("snapshots", help="snapshot JSONL path")
    check.add_argument("--base-url", required=True, help="vLLM server base URL (without /tokenize)")
    check.add_argument("--format", choices=("text", "json"), default="text")
    check.add_argument(
        "--fail-on-max-model-len-change",
        action="store_true",
        help="also fail when the serving context limit differs from the snapshot",
    )
    add_http_options(check)

    diff = commands.add_parser("diff", help="compare two snapshots offline")
    diff.add_argument("expected", help="approved snapshot JSONL path")
    diff.add_argument("actual", help="new snapshot JSONL path")
    diff.add_argument("--format", choices=("text", "json"), default="text")
    diff.add_argument(
        "--fail-on-max-model-len-change",
        action="store_true",
        help="treat context-limit metadata changes as failures",
    )
    return parser


def add_http_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-key-env",
        metavar="NAME",
        help="environment variable containing the server API key (never stored)",
    )
    parser.add_argument(
        "--timeout", type=positive_float, default=30.0, help="request timeout in seconds"
    )
    parser.add_argument(
        "--proxy-from-env",
        action="store_true",
        help="route through standard HTTP_PROXY/HTTPS_PROXY environment settings",
    )


def atomic_write(path: str, content: bytes) -> None:
    target = Path(path)
    parent = target.parent
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            temporary_path = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    except OSError as exc:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
        raise PromptcaseError(f"cannot write {path}: {exc.strerror or exc}") from exc


def write_output(path: str, content: bytes) -> None:
    if path == "-":
        sys.stdout.buffer.write(content)
        return
    atomic_write(path, content)


def row_counts(findings: Sequence[Mapping[str, Any]]) -> tuple[int, int, int, int]:
    drifted_ids = {finding["id"] for finding in findings if finding["code"] == "TOKEN_DRIFT"}
    missing = sum(finding["code"] == "CASE_MISSING" for finding in findings)
    added = sum(finding["code"] == "CASE_ADDED" for finding in findings)
    metadata = sum(finding["code"] == "MAX_MODEL_LEN_CHANGED" for finding in findings)
    return len(drifted_ids), missing, added, metadata


def result_document(
    command: str, cases: int, findings: list[dict[str, Any]], errors: int = 0
) -> dict[str, Any]:
    drifted, missing, added, metadata = row_counts(findings)
    failures = drifted + missing + added + errors
    return {
        "schema_version": 1,
        "command": command,
        "ok": failures == 0,
        "summary": {
            "cases": cases,
            "drifted": drifted,
            "missing": missing,
            "added": added,
            "metadata_changed": metadata,
            "errors": errors,
        },
        "findings": findings,
    }


def render_json(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def format_token(value: Any) -> str:
    return "EOF" if value is None else str(value)


def render_text(document: Mapping[str, Any], *, label: str) -> str:
    summary = document["summary"]
    status = "PASS" if document["ok"] else "FAIL"
    lines = [
        f"promptcase {label} {status}",
        "cases={cases} drifted={drifted} missing={missing} added={added} "
        "metadata-changed={metadata_changed} errors={errors}".format(**summary),
    ]
    for finding in document["findings"]:
        code = finding["code"]
        case_id = finding["id"]
        if code == "TOKEN_DRIFT":
            lines.append(
                f"DRIFT {case_id}: token {finding['index']} changed "
                f"{format_token(finding['expected_token'])} -> "
                f"{format_token(finding['actual_token'])}; "
                f"count {finding['expected_count']} -> {finding['actual_count']}"
            )
        elif code == "MAX_MODEL_LEN_CHANGED":
            lines.append(
                f"INFO {case_id}: max_model_len {finding['expected']} -> {finding['actual']}"
            )
        elif code == "CASE_MISSING":
            lines.append(f"MISSING {case_id}: case is absent from the new snapshot")
        elif code == "CASE_ADDED":
            lines.append(f"ADDED {case_id}: case is absent from the approved snapshot")
        elif code == "REQUEST_FAILED":
            lines.append(f"ERROR {case_id}: {finding['message']}")
    return "\n".join(lines) + "\n"


def run_record(args: argparse.Namespace) -> int:
    cases = load_cases(args.requests)
    snapshots: list[dict[str, Any]] = []
    for case in cases:
        response = post_tokenize(
            args.base_url,
            case["request"],
            timeout=args.timeout,
            api_key_env=args.api_key_env,
            use_environment_proxy=args.proxy_from_env,
        )
        result = validate_result(response, case_id=case["id"], path="<server>", line_number=0)
        snapshots.append(make_snapshot(case, result))
    write_output(args.output, encode_jsonl(snapshots))
    if args.output != "-":
        print(f"recorded {len(snapshots)} case(s) in {args.output}", file=sys.stderr)
    return 0


def run_check(args: argparse.Namespace) -> int:
    approved = load_snapshots(args.snapshots)
    actual: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    errors = 0
    for row in approved:
        try:
            response = post_tokenize(
                args.base_url,
                row["request"],
                timeout=args.timeout,
                api_key_env=args.api_key_env,
                use_environment_proxy=args.proxy_from_env,
            )
            result = validate_result(response, case_id=row["id"], path="<server>", line_number=0)
            actual.append(make_snapshot({"id": row["id"], "request": row["request"]}, result))
        except PromptcaseError as exc:
            errors += 1
            findings.append({"code": "REQUEST_FAILED", "id": row["id"], "message": str(exc)})

    failed_ids = {finding["id"] for finding in findings}
    token_findings = [
        finding
        for finding in compare_snapshot_sets(approved, actual)
        if not (finding["code"] == "CASE_MISSING" and finding["id"] in failed_ids)
    ]
    findings.extend(token_findings)
    document = result_document("check", len(approved), findings, errors)
    if args.fail_on_max_model_len_change and document["summary"]["metadata_changed"]:
        document["ok"] = False
    else:
        document["ok"] = document["summary"]["drifted"] == 0 and errors == 0
    emit_report(document, args.format, label="check")
    return 2 if errors else (1 if not document["ok"] else 0)


def run_diff(args: argparse.Namespace) -> int:
    expected = load_snapshots(args.expected)
    actual = load_snapshots(args.actual)
    findings = compare_snapshot_sets(expected, actual)
    document = result_document("diff", len(expected), findings)
    if args.fail_on_max_model_len_change and document["summary"]["metadata_changed"]:
        document["ok"] = False
    else:
        document["ok"] = (
            document["summary"]["drifted"] == 0
            and document["summary"]["missing"] == 0
            and document["summary"]["added"] == 0
        )
    emit_report(document, args.format, label="diff")
    return 1 if not document["ok"] else 0


def emit_report(document: Mapping[str, Any], output_format: str, *, label: str) -> None:
    if output_format == "json":
        sys.stdout.buffer.write(render_json(document))
    else:
        sys.stdout.write(render_text(document, label=label))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            return run_record(args)
        if args.command == "check":
            return run_check(args)
        if args.command == "diff":
            return run_diff(args)
        parser.error(f"unknown command {args.command!r}")
    except PromptcaseError as exc:
        print(f"promptcase: error: {exc}", file=sys.stderr)
        return 2
    return 2
