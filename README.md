# Promptcase

[![CI](https://github.com/thisbejim/promptcase/actions/workflows/ci.yml/badge.svg)](https://github.com/thisbejim/promptcase/actions/workflows/ci.yml)

**Record and regression-test the exact token IDs a vLLM server produces from your chat requests.**

A chat request can be valid JSON and still turn into the wrong model input. Chat templates and server-side content handling decide how messages, tool results, special tokens, and generation prefixes become token IDs. Promptcase stores those IDs as replayable fixtures so a template, tokenizer, or serving update can be reviewed before it changes model behavior.

## Problem

Recent inference bugs have silently changed a tool result or added a duplicate BOS token during prompt rendering. Those failures are difficult to spot in a response or API schema check; inspecting the server's token IDs reveals what actually reached the model.

Promptcase automates that inspection for a team's own requests. It calls vLLM's chat-aware `/tokenize` endpoint, never asks the model to generate text, and reports the first changed token when a later server version renders a fixture differently.

## Quick start

Install Promptcase:

```bash
python -m pip install "promptcase @ git+https://github.com/thisbejim/promptcase.git@v0.1.0"
```

Create `requests.jsonl` with one request per line:

```json
{"id":"tool-result","request":{"model":"my-model","messages":[{"role":"user","content":"What is the weather?"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":"15°C, cloudy"}],"tools":[{"type":"function","function":{"name":"weather","description":"Get weather","parameters":{"type":"object","properties":{}}}}],"add_generation_prompt":true}}
```

With a local vLLM server already running:

```bash
promptcase record requests.jsonl \
  --base-url http://127.0.0.1:8000 \
  --output prompts.snapshot.jsonl
promptcase check prompts.snapshot.jsonl --base-url http://127.0.0.1:8000
```

Commit the request and snapshot files. Run `check` in CI against the server image or model bundle you want to release. `record` makes a baseline; `check` compares it with the current server; `diff` compares two snapshots without a server.

For an authenticated endpoint, store the key in an environment variable and name it explicitly:

```bash
export VLLM_API_KEY="..."
promptcase check prompts.snapshot.jsonl \
  --base-url https://inference.internal.example \
  --api-key-env VLLM_API_KEY
```

The configured base URL receives each full request payload. Use a trusted endpoint. Prompts and tool definitions in the snapshot are sensitive data; reports omit their contents.

## Example output

```text
promptcase check FAIL
cases=1 drifted=1 missing=0 added=0 metadata-changed=0 errors=0
DRIFT tool-result: token 173 changed 99082 -> 27; count 181 -> 176
```

Token IDs are shown because they identify the first changed location without printing prompt text or token strings. Context-limit changes are informational by default; `--fail-on-max-model-len-change` makes them fail CI too.

The snapshots under [`examples/`](examples/) use synthetic token IDs for an offline demonstration. They do not claim to be output from a real model. The compact example command is:

```bash
promptcase diff examples/baseline.jsonl examples/changed.jsonl
```

## Supported input

Each request row has an `id` and a `request` object. Promptcase forwards these vLLM tokenization fields: `model`, `messages`, `tools`, `add_generation_prompt`, `add_special_tokens`, `chat_template`, `chat_template_kwargs`, `continue_final_message`, and `mm_processor_kwargs`. Common generation settings such as `temperature`, `max_tokens`, `stream`, and `seed` are accepted and omitted because they do not affect tokenization. Options such as `tool_choice` and `response_format` are rejected because the tokenize endpoint does not accept them. Unknown request fields fail with an explanation so a fixture cannot silently lose an unsupported template setting.

It calls the documented vLLM `POST /tokenize` route and checks the `count`, `tokens`, and `max_model_len` response. A base URL ending in `/v1` is normalized to the server root. Text, tool calls, custom template options, and vLLM-supported multimodal request fields are passed through. Promptcase does not inspect image or audio meaning; it checks only the tokenization result returned by the server.

## Commands and exit codes

```text
promptcase record REQUESTS.jsonl --base-url URL [--output SNAPSHOT.jsonl]
promptcase check SNAPSHOT.jsonl --base-url URL [--format text|json]
promptcase diff EXPECTED.jsonl ACTUAL.jsonl [--format text|json]
```

- `0`: tokens match, or an offline diff contains no case/token changes.
- `1`: token drift or case addition/removal was found.
- `2`: invalid input, invalid server response, or request failure.

`MAX_MODEL_LEN_CHANGED` is a reported metadata finding. Use `--fail-on-max-model-len-change` to make it fail.

## Why Promptcase

Hugging Face's `apply_chat_template` is a good reference for local rendering. vLLM's `/tokenize` endpoint is better when the question is what the deployed server actually produced, including its selected tokenizer, template, and request parsing. Promptcase turns that endpoint into a committed test corpus and CI gate, so each team does not have to write its own JSONL replay and token-array diff script.

Promptcase is intentionally scoped to vLLM's documented tokenization route. It does not claim that hosted model APIs expose their rendered prompt, certify OpenAI compatibility, or prove that an approved template is semantically correct.

## Privacy and network behavior

- No telemetry, analytics, ads, or background network calls.
- `diff` and snapshot validation work offline.
- `record` and `check` send each fixture to the `--base-url` you provide. They do not call a model completion endpoint.
- Requests go directly to that host by default. Use `--proxy-from-env` to route through `HTTP_PROXY` or `HTTPS_PROXY` settings. Redirects are not followed.
- Authentication is optional and read only from the environment variable named by `--api-key-env`; credentials are never written to snapshots or reports.
- Snapshots include the full request and token IDs to enable replay. Protect them like prompts, tool definitions, and other private model inputs.
- Reports contain case IDs, counts, changed token IDs, and token positions, but not request text or token strings.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check .
ruff format --check .
mypy
python -m build
```

The test suite uses a local fake tokenization server. It needs no GPU, model download, API key, paid service, or network. See [`docs/product-spec.md`](docs/product-spec.md) and [`docs/research.md`](docs/research.md) for scope and evidence.

## License

MIT. See [`LICENSE`](LICENSE).
