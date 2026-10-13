# Agent notes

## Implementing a new benchmark

Load the project skill **implement-benchmark** before writing code:

- Canonical: [`.agents/skills/implement-benchmark/`](.agents/skills/implement-benchmark/)
- Also linked at `.cursor/skills/implement-benchmark/` and `.claude/skills/implement-benchmark/`

Human checklist: [`python_package/docs/adding-benchmark.md`](python_package/docs/adding-benchmark.md).

New benchmarks subclass `BaseBenchmark`, are wired through `create_benchmark`, and run with `heams run`.
