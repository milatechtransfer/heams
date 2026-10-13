# Anti-patterns

If you see X in the codebase or are about to do Y, stop and follow the correct pattern.

## Layering and scope

| Don’t | Do |
|-------|-----|
| Edit `experiment_orchestration` / `orchestrator` so a new benchmark writes Markdown / `summary.json` | Emit `RunResult` to `runs.jsonl` only; put Markdown in package `report.py` and register `_ROLLUP_RENDERERS`; rollup reads artifacts, never calls `create_benchmark` |
| Drive-by refactor of shared `benchmarks/__init__.py`, `evaluation/__init__.py`, or global TYPE_CHECKING | Touch only what this suite needs; match Locomo lazy-export pattern; leave unrelated cleanup to a dedicated change |
| Route source records through `Scenario` then rebuild `BenchmarkSample` | Convert with a package helper like Locomo `record_to_benchmark_sample`; keep loading inside the benchmark package |
| Document that `heams run` writes `_rollup/...` | Rollups are offline/manual (`write_benchmark_rollup_reports`); do not claim CLI auto-rollup until it exists |
| Invent a parallel runner or CLI | `BaseBenchmark` + `create_benchmark`, run with `uv run heams run --config configs/runs/… --output-dir …` from `python_package/` |

## Config, datasets, materialization

| Don’t | Do |
|-------|-----|
| `_DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[N] / "datasets" / …` | `_DEFAULT_DATASET_DIR = settings.datasets_dir / "<name>"` |
| Smoke and full datasets share one cache path so smoke poisons full runs | Distinct dirs or explicit YAML sources; document which config downloads what |
| Prompt strings with `{placeholders}` and no validator | Fail at config validation if required placeholders are missing/extra |
| Convert every fixture then `samples = samples[:limit]` | Apply `limit` to records/fixtures **before** conversion/load |
| Wrap all materialization errors as “download failed” | Separate download/IO failures from per-record validation; soft-skip only reachable paths |
| Soft-skip loop after a batch-level crash already aborted | Structure try/except so per-record skips can run |
| Shipped `*_smoke.yaml` with `limit: 5` (or anything > 1) | `limit: 1`; assert in `test_shipped_yaml_configs_validate` |

## Samples, evaluators, tests, docs

| Don’t | Do |
|-------|-----|
| Docstring / docs say “3 samples per record” when code emits 1 sample × 3 stimuli | Match wording to `BenchmarkSample` cardinality |
| Treat partially missing evaluation markers as empty lists | Required markers: fail if any required key is missing |
| On unparsable judge output, default the sensitive metric to “safe” / non-leak | Fail closed when underreporting is dangerous (e.g. leakage) |
| Docs say “downloads BEAM” but smoke YAML points at a static JSON fixture (or the reverse) | Align smoke config text, YAML, and error messages |
| Unconditionally `import huggingface_hub` in unit tests | Optional dep: guard ImportError; loaders raise a friendly error |
| Stub evaluation record keys that only exist in tests (`:ci_summary` vs production `:summary`) | Use production key shapes in fixtures/stubs |
| Put agent instructions / memory strategy / model ids in benchmark Python | Component YAML under `configs/agents/`, `configs/memory_systems/`, `configs/benchmarks/` |
| Hard-code policy prompt defaults in Python | Required YAML fields (Locomo pattern) |
| Download datasets inside unit tests | One-record local fixture under `tests/fixtures/…` |
| Delete or ignore `.gitignore` while adding run configs that write `results/` | Keep ignore rules; never commit run artifacts |

## Quick “grep red flags”

If your diff touches these for a *new benchmark*, you are probably wrong:

- `src/heams/orchestration/`
- `Path(__file__).resolve().parents`
