---
name: implement-benchmark
description: >-
  Guides implementing a new HEAMS benchmark on the config-driven BaseBenchmark +
  create_benchmark + heams run path. Use when adding a benchmark, porting an eval
  suite, wiring a new type under configs/benchmarks/, or integrating Locomo-like,
  STALE, CI, Transparency, or EvoMemory workloads. Enforces reuse of existing
  packages and keeps orchestration code untouched.
---

# Implement a new benchmark (reuse-first)

Canonical path: `.agents/skills/implement-benchmark/` (Agent Skills). Also
discoverable via symlinks at `.cursor/skills/implement-benchmark/` (Cursor) and
`.claude/skills/implement-benchmark/` (Claude Code). Copilot loads
`.agents/skills/` project skills. Invoke as `/implement-benchmark` when supported.

Human checklist (do not restate here):
[`python_package/docs/adding-benchmark.md`](../../../python_package/docs/adding-benchmark.md)

Progressive disclosure:
- Shape → clone package → tests → file list: [exemplar-map.md](exemplar-map.md)
- Common don’ts grounded in current codebase practice: [anti-patterns.md](anti-patterns.md)

Put API keys (`OPENAI_API_KEY`, and `HF_TOKEN` for gated datasets) in the root `.env` before any `heams run`.

## Workflow

Copy and track:

```
Task progress:
- [ ] 1. Classify workload shape
- [ ] 2. Pick nearest exemplar (exemplar-map.md)
- [ ] 3. Read exemplar package + adding-benchmark.md
- [ ] 4. Reuse ladder (evaluator / env / rollup)
- [ ] 5. Clone package layout; wire factory only
- [ ] 6. Fixture test + smoke YAML (limit: 1) + heams run
- [ ] 7. Docs page + zensical.yml / docs/index.md / root README.md links
```

### 1–2. Classify, then clone

| Shape | Clone |
|-------|--------|
| Single-turn QA over primed conversation history | `locomo/` (or `long_mem_eval/` if judge lives under `evaluation/`) |
| External dataset needing an adapter | `transparency/` (+ `beam.py` pattern) |
| Multi-stimulus / environment tasks | `evomemory/` |
| Disclosure / policy / multi-judge | `contextual_integrity/` |
| Multi-probe single scenario + HF (or similar) dataset | `stale/` |
| Memory-poisoning / adversarial priming | `mpbench/` |
| Erasure / editability tasks | `erasability_editability/` |
| Long conversations primed once, rubric/nugget judge | `beam/` |
| Memory operations only (no agent turn), per-fact pass/fail | `retention_policy/` |
| Interactive environment / tool tasks | **Stop and ask**: no current benchmark uses a task environment |

Open first (under `python_package/`): exemplar `config.py` / `samples.py` / `benchmark.py`, plus `src/heams/benchmarks/factory.py` and `docs/adding-benchmark.md`.

### 3–4. Reuse ladder

Before writing new code:

1. **Evaluator:** `TaskQAEvaluatorConfig` → existing judges under `evaluation/` or sibling packages → only then new `@register` evaluator + `EvaluatorConfig` union.
2. **Environment:** prefer `scenario_priming.memory` / env priming (EvoMemory); do not invent a new executor.
3. **Reporting:** ship `runs.jsonl` first. Register a rollup renderer in `heams/reporting/benchmark_rollup.py` only if Markdown is required; read artifacts only — never call `create_benchmark` from rollup.

### 5. Implement

Clone Locomo-style layout: `config.py`, `dataset.py`, `samples.py`, `benchmark.py` (+ optional `judge.py` / `report.py`).

Direct `record → BenchmarkSample` (e.g. Locomo `record_to_benchmark_sample`). Do not route through `Scenario` then rebuild.

Wire exactly:
0. `report_info: ClassVar[BenchmarkInfo]` on the config class (`heams.benchmarks.info`): display name, `ReportDimension`, the judge tags naming criteria (`criterion_tags`), `context_tokens`, and `variant`/`scope` hooks if the config has splits or partial selections. Reports and `heams browse` read only this; never add benchmark names to reporting or UI code.
1. `BenchmarkConfig` union + `create_benchmark` branch in `factory.py`
2. Re-exports from package `__init__.py` and `benchmarks/__init__.py` (match Locomo lazy-export pattern)
3. If new evaluator: register + import in `evaluation/factory.py` + union
4. `configs/benchmarks/testing/<name>_smoke.yaml` + `configs/runs/testing/<name>_smoke.yaml`

Defaults: `_DEFAULT_DATASET_DIR = settings.datasets_dir / "<name>"`. Prompts/policy in YAML (`extra="forbid"`); validate template placeholders at config load.

### 6. Prove

- One-record local fixture; **no network download in unit tests**
- Assert factory validation/dispatch; mirror nearest test file from exemplar-map
- Shipped smoke YAML: `limit: 1` (assert in `test_shipped_yaml_configs_validate` when present)
- From `python_package/`: `uv run heams run --config configs/runs/testing/<name>_smoke.yaml --output-dir results/<name>_smoke`

### 7. Document

Add `docs/benchmarks/<name>.md`; link from `zensical.yml`, `docs/index.md` and the benchmark table in the root `README.md`. Smoke config and docs must agree (fixture vs download).

## Hard stops

1. Do **not** edit `orchestrator` / `experiment_orchestration` to add a benchmark (reporting is offline from `runs.jsonl`).
2. Do **not** bypass `create_benchmark`: every benchmark is a `BaseBenchmark` wired through the factory union.
3. Do **not** put agent/memory policy on samples (`to_scenario()` keeps agents/memory empty; experiment YAML binds them).
4. Do **not** default datasets via `Path(__file__).parents[…]` — use `settings.datasets_dir`.
5. Do **not** apply `limit` after converting the full dataset — slice before conversion/load.
6. Do **not** invent Scenario round-trips for sample materialization.
7. Do **not** drive-by refactor shared `__init__.py` / evaluation exports / TYPE_CHECKING for other suites.
8. Do **not** add CLI flags; comparisons = multiple experiments in run YAML.

## Minimal checklist before PR

- [ ] Nearest exemplar cloned; factory union + dispatch updated
- [ ] `report_info` declared on the config (`tests/benchmarks/test_info.py` passes)
- [ ] `settings.datasets_dir` default; early `limit`; soft-skips reachable
- [ ] No orchestrator changes; rollup only if needed (artifact replay)
- [ ] Smoke `limit: 1` + fixture test + docs/nav links
- [ ] PR scoped to this benchmark (no global cleanup)
