# Product specification

## Target developer

Inference and model-platform engineers serving Hugging Face chat models through vLLM's OpenAI-compatible endpoint. This also fits research engineers who maintain custom chat templates and need a cheap regression check before loading a checkpoint or running model evaluations.

## Problem

The request's `messages` array is not the model input. A server applies a model-specific chat template, converts roles, tool calls, tool results, special tokens, and generation prefixes into token IDs, then sends those IDs to inference. A malformed template or server-side content-format conversion can silently remove or alter parts of the conversation while the HTTP request still succeeds.

## Evidence

- A vLLM report showed that a tool result was present in a valid Chat Completions request but missing from the actual prompt after template rendering. The report used `/tokenize` and `/detokenize` to expose the lost content: [vLLM issue #39614](https://github.com/vllm-project/vllm/issues/39614).
- A separate vLLM report found duplicated BOS tokens between chat-template rendering and tokenization: [vLLM issue #2012](https://github.com/vllm-project/vllm/issues/2012).
- A Text Generation Inference request explained that its `/tokenize` endpoint did not apply the chat template, making the resulting count differ from `/v1/chat/completions`: [TGI issue #1706](https://github.com/huggingface/text-generation-inference/issues/1706).
- Hugging Face's documentation describes chat templates as the model-specific conversion from structured messages to the formatted prompt and token IDs: [Transformers chat templating](https://huggingface.co/docs/transformers/main/en/chat_templating).
- vLLM documents a chat-aware `/tokenize` request that returns the exact token IDs, count, token strings, and maximum model length produced by its serving tokenizer: [vLLM tokenization protocol](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/serve/tokenize/protocol/).

These sources show repeated failures across separate serving stacks and model families. Current repro steps are hand-built requests plus manual inspection of raw tokenization endpoints. The hidden transformation sits between a well-formed API request and the model input, so validating only the request schema does not catch these failures.

## Existing workflow and alternatives

- **Manual calls to vLLM `/tokenize` and `/detokenize`:** expose the server's actual token IDs and are useful for one-off debugging. They do not provide a fixture format, snapshots, CI checks, or a stable token-level regression report.
- **Hugging Face `apply_chat_template`:** gives a reference renderer through Python. It requires a handwritten script and may not reproduce a deployed server's engine-specific content normalization or tokenizer version.
- **TGI `/tokenize`:** provides a useful text tokenizer endpoint, but the cited request documents that its path did not apply chat templates. It cannot serve as a server-rendered chat golden for that use case.
- **vLLM and llama.cpp regression suites:** test their own implementations and model-specific cases. They are valuable upstream coverage, but application teams still need fixtures for their own requests and selected model bundle.
- **vLLM Bench:** has local Hugging Face and server-side tokenizer support and checks generated benchmark prompt lengths against the server on first use. It is designed for benchmark sizing and load tests, not token-ID baselines and exact per-case regression diffs.
- **Prime Intellect `renderers`:** provides a programmable rendering library for training and inference integrations. Its focus is application-side rendering, response parsing, and rollout bridging rather than committing and replaying server-rendered token snapshots from a JSONL request corpus.
- **The existing `token-bundle-audit` project in this workspace:** audits tokenizer metadata and chat-template portability without executing templates or contacting a model server. Promptcase addresses runtime output and request-specific regressions.

No alternative found in this research offered a small, local-first CLI that records a team's own chat payloads as token-ID fixtures, replays them against the serving tokenizer, and reports the first changed token in CI.

## Candidate selection

Three serious alternatives and the selected project were scored against all dimensions in the goal. Scores are ordered as: pain, frequency, audience size, demand evidence, repeated reinvention, alternative weakness, improvement, advanced-AI applicability, OpenAI/Meta/xAI-style ecosystem relevance, standalone usefulness, local-first advantage, discoverability, time to first value, feasibility, maintainability, testability, deterministic correctness.

| Candidate | 17 scores in the order above | Main deductions | Decision |
|---|---|---|---|
| Chat-template runtime snapshots | 8, 7, 8, 8, 7, 8, 8, 9, 9, 9, 9, 7, 8, 9, 8, 9, 9 | Works with vLLM's documented route only; server/model-specific outputs need local fixtures | Build: visible failures, small deterministic surface, and no model generation needed |
| Prompt-prefix cache diagnosis | 9, 7, 9, 8, 7, 5, 6, 9, 9, 8, 8, 7, 5, 5, 5, 7, 6 | Similar CI analyzers and cache simulators exist; exact results require provider/runtime tokenization and cache-policy assumptions | Reject: the expected improvement over existing tools does not clear the quality gate |
| Statistical significance for eval diffs | 8, 8, 9, 9, 8, 4, 5, 9, 9, 9, 9, 8, 8, 9, 7, 9, 9 | Several focused packages already provide paired tests, bootstrap intervals, and sample-size planning; lm-evaluation-harness reports bootstrap standard errors | Reject: the workflow is already well served by newer focused tools |
| Embeddings endpoint conformance | 7, 7, 8, 7, 6, 6, 6, 8, 8, 8, 8, 7, 7, 8, 7, 9, 9 | A broad OpenAI-compatible endpoint tester includes embedding profiles; adjacent local tools already cover request and exported-vector contracts | Reject: the remaining distinction is too narrow for a strong standalone improvement |

## Product thesis

For inference engineers, Promptcase turns real chat requests into committed token-ID snapshots and checks them against the deployed vLLM tokenizer, catching template and preprocessing drift before it changes model behavior; unlike raw endpoint probing, each case is replayable and reviewable in CI without generating a completion.

## Core workflow

```text
chat request JSONL + local vLLM tokenizer
  ↓  promptcase record
versioned snapshot JSONL with the actual token IDs
  ↓  commit snapshot and request fixture
new server image + promptcase check
  ↓
PASS, or a report with the first token index that changed
```

Snapshots preserve the input request so they can be replayed. Reports do not print prompts, tool arguments, or token strings by default.

## Interface

Python 3.10+ CLI with three operations:

- `record`: call the explicitly selected vLLM `/tokenize` endpoint and write a stable JSONL snapshot.
- `check`: replay snapshots against the endpoint and fail on token-ID drift.
- `diff`: compare two snapshots entirely offline and identify missing cases and first token divergence.

It uses only the Python standard library at runtime. No model weights are loaded by the CLI.

## Offline story

Validation and `diff` work offline. Examples include synthetic token snapshots and do not require a GPU, model download, API key, or server. `record` and `check` require a vLLM server the user already operates; the model server handles tokenization but no generation is requested.

## Integration story

Promptcase accepts a documented subset of vLLM's tokenization request: model, messages, tools, template options, and multimodal processor options. It does not claim that an OpenAI-hosted API exposes tokenization or that every OpenAI-compatible server implements the same route. A configured local vLLM endpoint is the primary integration. Other runtimes can be added only with a documented tokenization contract and fixture coverage.

## Non-goals

- Generating completions or calling model-provider APIs.
- Loading weights, downloading tokenizers, or executing model-repository code.
- Proving that a response is good or that a template is semantically correct. Snapshots only assert the exact tokenization a team chose to approve.
- Claiming universal compatibility across inference engines.
- Replacing upstream vLLM template, tokenizer, or API tests.

## Privacy and security

The CLI has no telemetry, tracking, or background network access. `record` and `check` send the stored request to the exact base URL the user supplies. Snapshot files contain their request payloads and token IDs, so they must be treated like the prompts and tool definitions they preserve. Standard output reports IDs, counts, and token indexes only. API keys are read from an explicitly named environment variable and are never written to snapshots.

## Final pre-build challenge

An inference engineer would install this to keep a tool-use or structured prompt fixture stable across vLLM, Transformers, or model-bundle updates without spending time on generation. They would not dismiss it as another shallow AI repository because it checks the server's actual token IDs for their own requests, uses fixture-first tests, makes no quality claims, and has a narrow documented protocol boundary.
