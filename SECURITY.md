# Security policy

Please report suspected security issues privately through GitHub's security advisory feature for this repository.

Promptcase parses untrusted JSONL as data and does not execute model output, import model code, or fetch model assets. `record` and `check` send request fixtures to the explicitly supplied endpoint; do not point them at an untrusted service with sensitive prompts. Snapshot files contain the original request payload and should be protected accordingly.
