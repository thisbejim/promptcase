# Contributing

Bug reports and focused pull requests are welcome. Keep runtime dependencies at zero, preserve offline tests, and document the exact vLLM protocol fields a change relies on.

Before opening a pull request, run:

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
ruff format --check .
mypy
```

For an added protocol behavior, include a local fixture test for normal and malformed server responses. Do not include real prompts, credentials, or private traces in fixtures.
