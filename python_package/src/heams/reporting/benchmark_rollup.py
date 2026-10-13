"""Artifact-only benchmark rollup shared by typed benchmark families."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import yaml

from heams.memory_systems.factory import MemorySystemVariant
from heams.results import RunResult
from heams.scenarios.actions import UserStimulus
from heams.scenarios.base import Scenario

_ROLLUP_DIRNAME = "_rollup"

RollupRenderer = Callable[
    [Sequence[Scenario], dict[str, list[RunResult]], Sequence[str]],
    list[str],
]


def memory_label_from_variant_payload(payload: object) -> str:
    if not isinstance(payload, dict):
        return "memory"
    try:
        variant = MemorySystemVariant.model_validate(payload)
    except Exception:
        backend = payload.get("backend_config")
        if isinstance(backend, dict) and backend.get("type"):
            return str(backend["type"])
        return str(payload.get("id") or "memory")
    return variant.id or variant.backend_type


def load_experiment_config(experiment_dir: Path) -> dict[str, object] | None:
    artifacts = sorted(experiment_dir.glob("experiment_config.*.yaml"))
    if not artifacts:
        return None
    payload = yaml.safe_load(artifacts[-1].read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def load_runs(jsonl_path: Path) -> list[RunResult]:
    runs: list[RunResult] = []
    with jsonl_path.open(encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            runs.append(RunResult.model_validate_json(line))
    return runs


def iter_experiment_dirs_for_type(
    results_dir: Path,
    benchmark_type: str,
) -> Iterator[tuple[Path, dict[str, object], str]]:
    for experiment_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if experiment_dir.name == _ROLLUP_DIRNAME:
            continue
        if not (experiment_dir / "runs.jsonl").is_file():
            continue
        experiment_config = load_experiment_config(experiment_dir)
        if experiment_config is None:
            continue
        benchmark_config = experiment_config.get("benchmark_config")
        if not isinstance(benchmark_config, dict):
            continue
        if benchmark_config.get("type") != benchmark_type:
            continue
        label = memory_label_from_variant_payload(
            experiment_config.get("memory_system_config")
        )
        yield experiment_dir, experiment_config, label


def reconstruct_rollup_inputs(
    results_dir: Path,
    benchmark_type: str,
) -> tuple[list[Scenario], dict[str, list[RunResult]], list[str]] | None:
    memory_labels: list[str] = []
    seen_labels: set[str] = set()
    for _, _, label in iter_experiment_dirs_for_type(results_dir, benchmark_type):
        if label not in seen_labels:
            seen_labels.add(label)
            memory_labels.append(label)
    if not memory_labels:
        return None

    label_to_index = {label: index for index, label in enumerate(memory_labels)}
    remapped_results: dict[str, list[RunResult]] = defaultdict(list)
    remapped_scenarios: dict[str, Scenario] = {}

    for experiment_dir, _, label in iter_experiment_dirs_for_type(
        results_dir, benchmark_type
    ):
        memory_index = label_to_index[label]
        for run in load_runs(experiment_dir / "runs.jsonl"):
            remapped_results[run.scenario_id].append(
                run.model_copy(update={"memory_index": memory_index})
            )
            if run.scenario_id not in remapped_scenarios:
                remapped_scenarios[run.scenario_id] = Scenario(
                    id=run.scenario_id,
                    name=run.scenario_id,
                    execution_sequence=[UserStimulus(content="q")],
                    agents=[],
                    memory_systems=[
                        MemorySystemVariant.model_validate(
                            {
                                "id": memory_name,
                                "backend_config": {
                                    "type": "no_memory",
                                    "name": memory_name,
                                    "top_k": 1,
                                },
                            }
                        )
                        for memory_name in memory_labels
                    ],
                    evaluators=[],
                )

    return list(remapped_scenarios.values()), dict(remapped_results), memory_labels


def write_typed_rollup_report(
    results_dir: Path,
    *,
    benchmark_type: str,
    render_sections: RollupRenderer,
    title: str | None = None,
) -> Path | None:
    if not results_dir.is_dir():
        return None
    reconstructed = reconstruct_rollup_inputs(results_dir, benchmark_type)
    if reconstructed is None:
        return None
    scenarios, results, memory_labels = reconstructed
    sections = render_sections(scenarios, results, memory_labels)
    if not sections:
        return None

    report_path = results_dir / _ROLLUP_DIRNAME / benchmark_type / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    heading = title or f"{benchmark_type} benchmark"
    body = [f"# {heading}", "", *sections]
    report_path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")
    return report_path


def _locomo_renderer(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    from heams.benchmarks.locomo.report import (
        render_locomo_detail_sections,
        render_locomo_metrics_summary,
    )

    sections = render_locomo_metrics_summary(scenarios, results, memory_labels)
    detail = render_locomo_detail_sections(scenarios, results)
    if sections and detail:
        return [*sections, "", *detail]
    return sections or detail


def _transparency_renderer(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    from heams.benchmarks.transparency.report import (
        render_transparency_rollup_sections,
    )

    return render_transparency_rollup_sections(scenarios, results, memory_labels)


def _stale_renderer(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    from heams.benchmarks.stale.report import render_stale_rollup_sections

    return render_stale_rollup_sections(scenarios, results, memory_labels)


def _ee_renderer(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    from heams.benchmarks.erasability_editability.report import (
        render_ee_rollup_sections,
    )

    return render_ee_rollup_sections(scenarios, results, memory_labels)


def _ci_renderer(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    from heams.benchmarks.contextual_integrity.report import render_ci_rollup_sections

    return render_ci_rollup_sections(scenarios, results, memory_labels)


def _mpbench_renderer(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    from heams.benchmarks.mpbench.report import (
        render_mpbench_detail_sections,
        render_mpbench_metrics_summary,
    )

    sections = render_mpbench_metrics_summary(scenarios, results, memory_labels)
    detail = render_mpbench_detail_sections(scenarios, results)
    if sections and detail:
        return [*sections, "", *detail]
    return sections or detail


def _beam_renderer(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    from heams.benchmarks.beam.report import (
        render_beam_detail_sections,
        render_beam_metrics_summary,
    )

    sections = render_beam_metrics_summary(scenarios, results, memory_labels)
    detail = render_beam_detail_sections(scenarios, results)
    if sections and detail:
        return [*sections, "", *detail]
    return sections or detail


def _retention_policy_renderer(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    from heams.benchmarks.retention_policy.report import (
        render_retention_policy_rollup_sections,
    )

    return render_retention_policy_rollup_sections(scenarios, results, memory_labels)


_ROLLUP_RENDERERS: dict[str, RollupRenderer] = {
    "locomo": _locomo_renderer,
    "transparency": _transparency_renderer,
    "stale": _stale_renderer,
    "ee": _ee_renderer,
    "ci": _ci_renderer,
    "mpbench": _mpbench_renderer,
    "beam": _beam_renderer,
    "retention_policy": _retention_policy_renderer,
}


def write_benchmark_rollup_reports(results_dir: Path) -> dict[str, Path]:
    """Write `_rollup/<type>/report.md` for each known typed benchmark present."""
    written: dict[str, Path] = {}
    if not results_dir.is_dir():
        return written
    for benchmark_type, renderer in _ROLLUP_RENDERERS.items():
        path = write_typed_rollup_report(
            results_dir,
            benchmark_type=benchmark_type,
            render_sections=renderer,
        )
        if path is not None:
            written[benchmark_type] = path
    return written


__all__ = [
    "iter_experiment_dirs_for_type",
    "load_experiment_config",
    "load_runs",
    "memory_label_from_variant_payload",
    "reconstruct_rollup_inputs",
    "write_benchmark_rollup_reports",
    "write_typed_rollup_report",
]
