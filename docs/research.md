# Opportunity research

Research conducted 2026-09-24. Public sources below were read before implementation. The focus is the recurring engineering job of proving what a chat server actually tokenizes after applying a model's template.

## Pain evidence

1. In [vLLM issue #39614](https://github.com/vllm-project/vllm/issues/39614), a successful chat request returned a response that behaved as if a tool result were absent. The report decoded the actual prompt and found the server had replaced the tool result with an empty tool wrapper. The request was valid at the API layer; only rendered-token inspection exposed the cause.
2. [vLLM issue #2012](https://github.com/vllm-project/vllm/issues/2012) describes chat-template rendering and tokenization both adding special tokens, yielding an unexpected duplicate BOS token.
3. In [TGI issue #1706](https://github.com/huggingface/text-generation-inference/issues/1706), a user explained that `/tokenize` without chat-template application was not the prompt submitted to `/v1/chat/completions`, so they could not reliably count or truncate the actual model input.
4. The official [Hugging Face chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating) explains that roles, separators, generation prompts, and special tokens are rendered according to the model's template and that `tokenize=True` is needed for exact token IDs.
5. The current [vLLM tokenization protocol](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/serve/tokenize/protocol/) exposes the serving tokenizer's chat-aware request fields and returns token IDs and token strings. This makes the hidden intermediate state inspectable without generating output.

The signal comes from separate bug and feature reports in two inference repositories, plus official format documentation. It is not based on star counts or a single request for a product.

## Alternatives and their fit

| Alternative | Strength | Gap for this job |
|---|---|---|
| vLLM `/tokenize` and `/detokenize` | Exact server-side token IDs; no completion generation | One-off endpoint calls; no snapshot corpus, offline diff, or CI exit status |
| [vLLM Bench](https://github.com/vllm-project/vllm-bench) | Loads local tokenizers or falls back to vLLM's `/tokenize`; verifies benchmark prompt lengths against the server | Its tokenizer integration validates benchmark sizing, not committed token-ID snapshots and first-divergence reports for a team's own chat cases |
| Hugging Face Transformers `apply_chat_template` | Local reference rendering and token IDs | Handwritten harness; does not necessarily reflect the deployed server's parser/content normalization |
| TGI `/tokenize` | Token-count utility without loading the application tokenizer | The cited issue states that chat-template application was absent from that endpoint at the time |
| Upstream engine parity tests | Strong engine-specific regression suites | Not directly reusable for app-specific request fixtures or release gates |
| Prime Intellect `renderers` | Rich library for rendering and parsing in RL/inference applications | Different integration objective; not a small golden-fixture CLI against a running serving tokenizer |
| Browser template playgrounds | Quick manual inspection | Not a deterministic local CI workflow; some depend on a hosted tokenizer service |

## Candidate screening

- **Prompt-prefix cache diagnosis:** clearly relevant and supported by vLLM cache docs and issues. Existing tools include [CacheSentry](https://github.com/PS4Emp/cachesentry), [vLLM Doctor](https://docs.vllm.doctor/rules/prefix-cache-efficiency/), and [RouteSim](https://github.com/InfraWhisperer/RouteSim). Provider-side tokenization and cache breakpoints also prevent a local structural diff from proving real cache hits. Rejected.
- **Evaluation uncertainty:** statistically sound paired comparisons are useful, but focused packages including [evalci](https://github.com/Shreyaskc/evalci), [abeval](https://github.com/mohammadi-hadi/abeval), [evalstats](https://github.com/ianarawjo/evalstats), and [compton-eval-analysis](https://github.com/ommiles/compton-eval-analysis) already provide bootstrap confidence intervals or paired testing; the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) also supports bootstrap standard errors. Rejected because a new general CLI would have to show a much more specific unmet workflow.
- **Embeddings endpoint conformance:** vLLM discussion and SDK issues provide concrete examples around batch order, dimensions, and usage, but current compatibility tools such as [openai-compatible-tester-cli](https://github.com/ibidathoillah/openai-compatible-tester-cli) cover embedding endpoints and this workspace already contains adjacent schema and vector-export projects. Rejected as insufficiently distinct.
- **Chat-template runtime snapshots:** reports from vLLM and TGI show specific correctness failures that are invisible in ordinary request validation. A small standard-library CLI can automate the exact investigative step developers already perform, against the real local tokenizer, with no generated output. Selected.

## Discoverability

Intended searches: “vLLM prompt token regression test”, “chat template token IDs diff”, “test vLLM chat template”, “inspect rendered LLM prompt tokens”, and “vLLM tokenize JSONL CI”. The name `promptcase` describes the fixture-based workflow without claiming provider certification.

## Competitive note

This is a focused complement to existing serving and tokenizer tooling. It is not a general prompt management product, tokenizer implementation, benchmark, or replacement for engine correctness suites. If upstream adds a fixture-replay regression CLI with the same request format and first-divergence report, the justification for maintaining this repository should be revisited.
