# Adding a new benchmark

New benchmarks use the config-driven factory path behind `heams run`. LoCoMo is
the current reference implementation under `heams/benchmarks/locomo/`.

!!! tip "Agent skill (Cursor, Claude Code, Copilot)"
    Contributor agents should load
    [`.agents/skills/implement-benchmark/`](../../.agents/skills/implement-benchmark/)
    (reuse gate, exemplar map, and anti-patterns). The same skill is
    linked from `.cursor/skills/` and `.claude/skills/` for tool discovery. This
    page remains the human step-by-step checklist; the skill does not replace it.

!!! note "Synchronous contract"
    `BenchmarkSample` is the workload unit. Build new benchmarks from
    `BenchmarkSample` and `Scenario`; all scenarios use the same ordered,
    synchronous executor.

## 1. Create a typed config

Create `src/heams/benchmarks/example/config.py`. The `type` literal is the YAML
discriminator. Put dataset paths, limits, evaluator settings, and benchmark
prompts here; agent and memory policy belong in their own component files.

```python
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from heams.evaluation.factory import TaskQAEvaluatorConfig


class ExampleBenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["example"] = "example"
    examples_file: Path
    limit: int | None = Field(default=None, ge=1)
    evaluator: TaskQAEvaluatorConfig
```

Use required YAML fields for prompts or policy text instead of hiding defaults
in Python. See `LocomoBenchmarkConfig` for the full pattern.

### Declare how the benchmark is reported

Next to `type`, declare the benchmark's report metadata as a
`BenchmarkInfo` (`heams.benchmarks.info`). Reports and `heams browse` read
display names, sections, ordering and criteria only from here, so nothing
else needs to change for the benchmark to appear correctly:

```python
from typing import ClassVar, Literal

from heams.benchmarks.info import BenchmarkInfo, ReportDimension


class ExampleBenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["example"] = "example"
    report_info: ClassVar[BenchmarkInfo] = BenchmarkInfo(
        display_name="Example",
        dimension=ReportDimension.PERFORMANCE,  # or PRIVACY
        criterion_tags=(("category",),),
        context_tokens=50_000,
    )
    ...
```

| Field | Meaning |
|---|---|
| `display_name` | Name shown in reports and the browser |
| `dimension` | Report section: `PERFORMANCE` or `PRIVACY` |
| `criterion_tags` | Judge-record tag groups that name a criterion, tried in order (`(("task_type", "phase"),)` gives "erase / pre") |
| `context_tokens` | Typical history per sample; performance benchmarks are listed by increasing size (`None` sorts last) |
| `variant` / `variant_context_tokens` | A config field that makes each value its own row, with its own size (BEAM's `split`) |
| `scope` | Whether a run covers the whole dataset (`FULL`), a capped sample (`SAMPLE`, the default when `limit` is set) or a different slice (`PARTIAL`, left out of cross-run tables, like a Retention Policy run on some policies) |
| `criterion_from_key` | Criterion of an audit key, for judges that write no report records |

Hooks receive the saved `benchmark_config` mapping, so they keep working on
results from older config versions. A judge can also name a record's
criterion directly with a `criterion` tag, which takes precedence over
`criterion_tags`. `tests/benchmarks/test_info.py` fails if a benchmark in the
`BenchmarkConfig` union does not declare `report_info`.

## 2. Emit concrete samples

Create `src/heams/benchmarks/example/benchmark.py`. A benchmark inherits
`BaseBenchmark`, materializes its dataset once, and exposes ordered
`BenchmarkSample` objects through `samples`. This minimal class expects JSON
shaped like `[{"id": "...", "question": "...", "answer": "..."}]`:

```python
import json
from collections.abc import Sequence

from heams.benchmarks.base import BaseBenchmark
from heams.benchmarks.example.config import ExampleBenchmarkConfig
from heams.benchmarks.sample import BenchmarkSample
from heams.scenarios.actions import UserStimulus


class ExampleBenchmark(BaseBenchmark):
    def __init__(self, config: ExampleBenchmarkConfig) -> None:
        self.config = config
        records = json.loads(config.examples_file.read_text(encoding="utf-8"))
        if config.limit is not None:
            records = records[: config.limit]
        self._samples = [
            BenchmarkSample(
                id=record["id"],
                name=record["question"],
                execution_sequence=[
                    UserStimulus(
                        content=record["question"],
                        step_evaluation_markers={
                            "expected_answer": record["answer"],
                        },
                    )
                ],
                evaluators=[config.evaluator],
            )
            for record in records
        ]
        self.skipped = []

    @property
    def samples(self) -> Sequence[BenchmarkSample]:
        return self._samples
```

`sample.to_scenario()` performs the mechanical projection. Experiment
orchestration then binds the resolved agent and memory system; benchmark code
must not replace either policy.

For a production adapter, validate external records at the dataset boundary and
record exclusions as `SkipReason`s. Keep one sample runnable in a fixture test.

## 3. Wire every factory surface

In `heams/benchmarks/factory.py`:

1. Import `ExampleBenchmark` and `ExampleBenchmarkConfig`.
2. Add `ExampleBenchmarkConfig` to the `BenchmarkConfig` discriminated union.
3. Add an `example` branch in `create_benchmark`.
4. Keep `create_benchmark` typed as `-> BaseBenchmark`.
5. Export the config (and benchmark class if public) through `__all__`.

Also re-export the new public types from `heams/benchmarks/__init__.py` and from
`heams/benchmarks/example/__init__.py`. Missing the union update makes YAML
validation fail before dispatch; missing the return annotation or exports leaves
the Python API inaccurate even if runtime dispatch works.

## 4. Wire a custom evaluator only when needed

Reuse `TaskQAEvaluatorConfig` for ordinary single-turn answer matching. If the
benchmark needs a new evaluator:

1. Define its typed config and registered evaluator implementation.
2. Import the implementation in `heams/evaluation/factory.py` so registration
   happens.
3. Add its config to the `EvaluatorConfig` discriminated union.
4. Re-export public evaluator types from `heams/evaluation/__init__.py`.

`create_evaluator` dispatches through the class registry using `config.type`;
adding an evaluator only to the benchmark config is insufficient.

## 5. Add component YAML

Add `configs/benchmarks/testing/example_smoke.yaml`:

```yaml
type: example
examples_file: datasets/example/seed.json
limit: 1
evaluator:
  type: task_qa
  match_mode: normalized_exact
```

Then add a run matrix in `configs/runs/testing/example_smoke.yaml`:

```yaml
run_name: example_smoke
experiments:
  - name: gpt4o_mini_no_memory_example
    agent_config_file: testing/gpt4o_mini_tool_use.yaml
    memory_system_config_file: testing/no_memory.yaml
    benchmark_config_file: testing/example_smoke.yaml
```

Run it from `python_package/`:

```sh
uv run heams run --config configs/runs/testing/example_smoke.yaml --output-dir results/example_smoke
```

The CLI needs no benchmark-specific flags.

## 6. Test the current path

Follow `tests/benchmarks/locomo/test_benchmark.py` for sample materialization and
`tests/orchestration/test_run_orchestration.py` for end-to-end wiring.

- Use a one-record local fixture; do not download a dataset in unit tests.
- Assert factory validation and dispatch.
- Assert `experiment_config.<run_id>.yaml`, `runs.jsonl`, and `run_status.json`.
- Use benchmark YAML `limit: 1` for a cheap smoke run.

Do not add tests for removed async paths or the legacy flag-driven reporting
path.

## 7. Add reporting deliberately

The current shared experiment path appends one `RunResult` per completed
scenario to `runs.jsonl` and writes `run_status.json`. `heams run` writes no
Markdown; `heams generate-report` then writes a `report.md` for every experiment,
of any benchmark, with sub-metric tables built from the tags on your
`EvaluationRecord`s. Tag records consistently (for example by category) and the
report needs no benchmark-specific code.

An optional per-benchmark rollup compares memory systems in one table. Rollups
are generated from Python (no `heams` command writes them) via
`write_benchmark_rollup_reports(results_dir)`, which groups experiment
directories by `benchmark_config.type` and writes `_rollup/<type>/report.md`
when a registered renderer exists (today: `locomo`, `transparency`, `stale`,
`ee`, `ci`, `mpbench`, `beam`, `retention_policy`). To add one, register a
type-keyed renderer in `heams.reporting.benchmark_rollup` that reads saved
artifacts only (never call `create_benchmark` from a rollup).
`heams/benchmarks/contextual_integrity/report.py` is a worked example: it exposes
`render_ci_rollup_sections(scenarios, results, memory_labels)` and is registered
as the `ci` renderer. Keep evaluator output generic (`EvaluationRecord` and
audit keys); the renderer owns Markdown layout.

In each `EvaluationRecord`, put the compared texts in `expected` and
`observed` and keep `reason` to the verdict (`extracted AIME answer '214',
expected '204'`), not a repr of the whole answer: the results browser shows
the three side by side.

## 8. Document and validate

Add `docs/benchmarks/example.md`, then link it from `zensical.yml`, `index.md`,
and `README.md`. Describe the dataset, evaluator, component YAML, and current
`heams run` command.

Build the documentation from `python_package/`:

```sh
uv run zensical build -f ../zensical.yml --strict
```
