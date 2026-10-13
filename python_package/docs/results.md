# Understanding results

Every `heams run` receives one exact run directory through `--output-dir`. One
invocation can contain many experiments; each experiment gets its own stable
directory below the run directory. Repeating the same command resumes from the
saved snapshots and `runs.jsonl` ledgers instead of creating a new run.
`--overwrite` is guarded: it clears only a directory whose HEAMS snapshots and
ledgers validate successfully; malformed or unrelated directories are left
untouched.

## Run-level artefacts

At `output_dir/`:

| File / directory | Contents |
|------------------|----------|
| `run_config.<run_id>.yaml` | Snapshot of the resolved run (experiments + paths) |
| `<experiment_name>/` | One stable directory per experiment |
| `submetrics.csv` | Written by `heams generate-report --detailed`: one row per tagged sub-metric score and efficiency metric, per sample |
| `_rollup/<type>/report.md` | Only if you generate the optional per-benchmark rollups (see below) |

`run_id` is generated at resolve time (`YYYYMMDD-HHMMSS-<8 hex chars>`).

## Per-experiment artefacts

Inside `<experiment_name>/`:

| File / directory | Contents |
|------------------|----------|
| `experiment_config.<run_id>.yaml` | Fully resolved agent, memory, and benchmark configs |
| `runs.jsonl` | One `RunResult` JSON object per completed scenario/sample combination (appended during the run) |
| `execution_traces.jsonl` | One ordered framework + agent execution trace per `RunResult` |
| `run_status.json` | Boolean `success`, runnable/skipped counts, optional `error` |
| `report.md` | Written by `heams generate-report`: metrics, a summary, and detail for up to `--max-examples` samples (10 by default) |
| `workspace/` | Scenario-scoped memory files and local vector-store directories |

`RunResult.trace_id` joins a sample to exactly one row in
`execution_traces.jsonl`, including after a resumed attempt. Each trace records
priming, scenario manipulations, memory operations, user stimuli, LLM calls,
and tool calls in timestamp order. Events expose structured arguments/results
and a display-ready `content` field. Failed agent calls retain partial
TinyAgent spans when the framework provides them. Priming wrappers store an
entry count only; per-entry retain/recall payloads remain on
`memory_operation` events (and on `RunResult.memory_ledger`).

Execution traces contain model-visible prompts, recalled memory, and tool
output. Treat them as private run artifacts; they may contain sensitive user
data. Size is typically on the order of `runs.jsonl` rather than a second copy
of the full priming corpus.

`heams generate-report` joins traces by `trace_id` when the sidecar is present
and adds a compact ordered sequence plus event/tool counts to each selected
sample detail. Reports generated from older directories without traces remain
unchanged.

The orchestrator derives storage paths per scenario and sample:

- All backends: `workspace/<scenario>-a0-m0-e0-s<sample>-memory/`

The current experiment path binds one agent, one memory system, and each
sample's evaluator, so the runtime combination key is `a0-m0-e0`. Persistent
backends own their layout within that directory; in-process/no-op backends leave
it empty until teardown removes it.

`heams run` writes no Markdown reports. Run `heams generate-report --results-dir
<output_dir>` afterwards to write a `report.md` in each experiment directory.

## Per-benchmark rollups (optional, Python only)

A rollup compares memory systems on one benchmark across all the experiments in
a run directory. No `heams` command writes rollups; generate them from saved
artifacts with `write_benchmark_rollup_reports(results_dir)`:

```sh
cd python_package
uv run python -c "from pathlib import Path; from heams.reporting.benchmark_rollup import write_benchmark_rollup_reports; print(write_benchmark_rollup_reports(Path('results/locomo_smoke')))"
```

The function:

1. Finds experiment directories that contain `runs.jsonl` and an
   `experiment_config.*.yaml`
2. Groups them by `benchmark_config.type`
3. Remaps memory labels from each experiment's resolved memory-system config
4. Writes `_rollup/<type>/report.md` for registered types (`locomo`,
   `transparency`, `stale`, `ee`, `ci`, `mpbench`, `beam`, `retention_policy`)

## Contextual Integrity rollup

The CI rollup is generated from saved artifacts by
`write_ci_rollup_report(results_dir)` (or as part of
`write_benchmark_rollup_reports`; no `heams` command writes it), which:

1. Finds experiment directories that contain `runs.jsonl` and a CI
   `benchmark_config`
2. Reads the memory label from each experiment's resolved memory-system config
3. Writes `_rollup/ci/report.md` with the leakage, coverage, and helpfulness
   table, one row per memory backend

To read it, open `_rollup/ci/report.md` and compare memory backends across the
leakage family (lower is better) and the coverage family (higher is better). What
each column means is defined in the
[Contextual Integrity evaluation section](benchmarks/contextual_integrity.md#evaluation).
Use each experiment's `runs.jsonl` for machine-readable follow-up.

## Browsing results

`heams browse --results-dir <dir>` opens a local web GUI over these artefacts:
a criterion table across memory systems (one run per benchmark, the
largest or the latest finished, computed from `runs.jsonl`), each experiment's samples,
and each sample's transcript, judge records, metrics and memory ledger. See
[Results browser](results-browser.md).

## Reading accuracy

For any benchmark:

1. Run `heams generate-report --results-dir <output_dir>` and open each
   experiment's `report.md`
2. Compare memory systems by comparing their experiments' reports, or generate
   the per-benchmark rollups above for one table per benchmark
3. Drill into the detail sections for failed keys and transcripts
4. Use each experiment's `runs.jsonl` for machine-readable follow-up

For LongMemEval:

1. Run `make run CONFIG=testing/long_mem_eval_smoke OUTPUT_DIR=results/long_mem_eval_smoke`, or execute the
   `long_mem_eval.yaml` entry from a per-memory full campaign matrix.
2. Run `heams generate-report` on the output directory and open each
   experiment's `report.md`, or read its `runs.jsonl`.
3. `report.md` has a sub-metric table per tag, including question type and
   abstention. LongMemEval has no per-benchmark rollup.

Higher accuracy is better for task correctness. Component-level metric tables
are emitted only when the artifact-backed report renderer supports them.

## Smoke check

```sh
cd python_package
make run CONFIG=smoke_test_benchmark OUTPUT_DIR=results/smoke_test_benchmark
uv run heams generate-report --results-dir results/smoke_test_benchmark
ls results/smoke_test_benchmark/*/report.md
```
