"""Small, bounded HTTP client for vLLM's chat-aware /tokenize route."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from promptcase.core import PromptcaseError

MAX_RESPONSE_BYTES = 50_000_000


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def tokenize_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise PromptcaseError("base URL must be an absolute http:// or https:// URL")
    if parsed.username or parsed.password:
        raise PromptcaseError("put credentials in an environment variable, not in the base URL")
    if parsed.query or parsed.fragment:
        raise PromptcaseError("base URL must not include a query string or fragment")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    prefix = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    return prefix.rstrip("/") + "/tokenize"


def post_tokenize(
    base_url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    api_key_env: str | None,
    use_environment_proxy: bool = False,
) -> dict[str, Any]:
    url = tokenize_url(base_url)
    body = dict(payload)
    body["return_token_strs"] = False
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    try:
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PromptcaseError(f"cannot encode tokenize request: {exc}") from exc
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        handlers: list[Any] = [_NoRedirect()]
        if not use_environment_proxy:
            handlers.append(urllib.request.ProxyHandler({}))
        opener = urllib.request.build_opener(*handlers)
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise PromptcaseError("tokenize response exceeds the 50 MB safety limit")
    except urllib.error.HTTPError as exc:
        exc.close()
        raise PromptcaseError(f"tokenize endpoint returned HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, TimeoutError):
            raise PromptcaseError(f"tokenize request timed out after {timeout:g} seconds") from None
        raise PromptcaseError(f"cannot reach tokenize endpoint: {reason}") from None
    except TimeoutError:
        raise PromptcaseError(f"tokenize request timed out after {timeout:g} seconds") from None
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromptcaseError(f"tokenize endpoint returned invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise PromptcaseError("tokenize endpoint response must be a JSON object")
    return result
