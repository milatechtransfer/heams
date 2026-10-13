# Copilot instructions

Project conventions for coding assistants are in [`AGENTS.md`](../AGENTS.md).

When adding or porting a benchmark (a new `type:` under `configs/benchmarks/`), load the project skill at `.agents/skills/implement-benchmark/` before writing code and follow its hard stops: no orchestrator edits for reporting, `settings.datasets_dir` for dataset defaults, and smoke configs with `limit: 1`. Human checklist: `python_package/docs/adding-benchmark.md`.
