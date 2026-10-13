"""Read-only index over a results directory.

Everything here works from saved artifacts only: experiments are found by
their ``runs.jsonl``, described by ``experiment_config.*.yaml`` and
``run_status.json``, and summarized from ``runs.jsonl`` itself. Run records
can be tens of megabytes each (full-context transcripts and memory ledgers),
so ``runs.jsonl`` is indexed once into byte offsets plus a compact summary per
record (including its criterion scores), and individual records are re-read
by seeking to their offset. Indexes are cached on disk and can be built on a
background thread while the GUI is already serving.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from heams.benchmarks.info import BenchmarkInfo, RunScope, benchmark_info
from heams.memory_systems.plugins import memory_display_names
from heams.reporting.criteria import sample_scores
from heams.results_browser.records import (
    LARGE_RECORD_BYTES,
    RecordFile,
    Span,
    Walker,
    summary_fields,
)
from heams.results_browser.traces import (
    TRACES_JSONL_FILENAME,
    event_span,
    index_traces,
    trace_overview,
)

RUNS_JSONL_FILENAME = "runs.jsonl"
SUBMETRIC_CSV_FILENAME = "submetrics.csv"
CACHE_DIRNAME = "_results_browser_cache"
#: Bump when the cached summary shape changes so stale caches are rebuilt.
_INDEX_VERSION = 3
_SKIP_DIR_PREFIX = "_"
_SKIP_DIR_NAMES = frozenset({"workspace", "__pycache__"})
_RUN_ID = re.compile(r"experiment_config\.((\d{8})-(\d{6})[^.]*)\.yaml$")
_QUESTION_PREVIEW_CHARS = 240


@dataclass(frozen=True)
class DisplayLimits:
    """Caps applied before a run record is sent to the browser."""

    max_chars: int = 20_000
    max_items: int = 200
    #: Tool I/O and ledger payloads can each hold ~1 MB of recalled context
    #: split across many small strings; above this many characters they are
    #: shown as a JSON preview instead of a full structure.
    max_block_chars: int = 50_000
    block_preview_chars: int = 4_000


_DEFAULT_LIMITS = DisplayLimits()


@dataclass
class Experiment:
    id: str
    path: Path
    name: str
    #: Benchmark config ``type``.
    benchmark: str
    #: Memory system variant id (or backend type when the variant has none).
    memory_system: str
    agent_model: str
    status: str
    config_path: Path | None
    runs_path: Path
    #: Run directory relative to the browsed root (where the run came from).
    run_dir: str = ""
    run_id: str = ""
    #: ``YYYY-MM-DD HH:MM`` the run started, from its ``run_id``.
    started: str = ""
    #: Samples the run scheduled, from ``run_status.json`` once it finished.
    planned_samples: int | None = None
    #: Samples written to ``runs.jsonl`` so far (finished or not).
    recorded_samples: int = 0
    #: ``benchmark_config.limit``; ``None`` means no cap.
    limit: int | None = None
    #: Config variant reported as its own benchmark (``BenchmarkInfo.variant``).
    variant: str | None = None
    #: Dataset coverage, as the benchmark declares it for this config.
    scope: RunScope = RunScope.FULL
    #: ``memory_system_config.backend_config.type``.
    memory_backend: str = ""
    #: Display names resolved from benchmark and plugin declarations.
    benchmark_label: str = ""
    memory_label: str = ""
    has_submetrics: bool = False
    #: ``execution_traces.jsonl`` next to ``runs.jsonl``, when the run wrote one.
    traces_path: Path | None = None

    @property
    def benchmark_key(self) -> str:
        """Benchmark row identity: the type, plus its variant if it has one."""
        return f"{self.benchmark}:{self.variant}" if self.variant else self.benchmark

    @property
    def info(self) -> BenchmarkInfo | None:
        return benchmark_info(self.benchmark)

    @property
    def subset(self) -> bool:
        """Whether the run covers only part of its dataset."""
        return self.scope is not RunScope.FULL

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "benchmark": self.benchmark_key,
            "benchmark_type": self.benchmark,
            "benchmark_label": self.benchmark_label,
            "variant": self.variant,
            "memory_system": self.memory_system,
            "memory_label": self.memory_label,
            "agent_model": self.agent_model,
            "status": self.status,
            "run_dir": self.run_dir,
            "run_id": self.run_id,
            "started": self.started,
            "samples": self.recorded_samples,
            "planned_samples": self.planned_samples,
            "limit": self.limit,
            "scope": self.scope.value,
            "subset": self.subset,
            "runs_bytes": self.runs_path.stat().st_size,
            "has_submetrics": self.has_submetrics,
            "has_traces": self.traces_path is not None,
        }


@dataclass
class RunIndex:
    signature: list[int]
    offsets: list[int]
    summaries: list[dict[str, Any]]


def _skip(root: Path, path: Path) -> bool:
    parts = path.relative_to(root).parts[:-1]
    return any(
        part.startswith(_SKIP_DIR_PREFIX) or part in _SKIP_DIR_NAMES for part in parts
    )


def _benchmark_identity(
    benchmark_config: dict[str, Any],
) -> tuple[str, str | None, RunScope, str]:
    """``(type, variant, scope, label)`` as the benchmark declares them."""
    benchmark = str(benchmark_config.get("type") or "unknown")
    info = benchmark_info(benchmark)
    if info is None:
        # A benchmark this version does not declare: shown under its type.
        limited = benchmark_config.get("limit") is not None
        return benchmark, None, RunScope.SAMPLE if limited else RunScope.FULL, benchmark
    variant = info.variant_of(benchmark_config)
    return benchmark, variant, info.scope(benchmark_config), info.label(variant)


def _memory_id(memory_config: dict[str, Any]) -> str:
    """Memory system id as saved: the variant ``id``, else the backend type.

    Read from the saved config rather than validated, so results stay
    readable when the plugin that produced them is not installed.
    """
    backend = memory_config.get("backend_config") or {}
    return str(memory_config.get("id") or backend.get("type") or "memory")


def memory_labels(experiments: Iterable[Experiment]) -> dict[str, str]:
    """Display name of each memory system id, from plugin declarations.

    An id is named after its plugin's ``display_name``, or kept as is when
    the plugin declares none or is not installed. See
    :func:`distinct_labels` for telling apart configurations of one plugin.
    """
    declared = memory_display_names()
    return {
        e.memory_system: declared.get(e.memory_backend, e.memory_system)
        for e in experiments
    }


def distinct_labels(labels: dict[str, str]) -> dict[str, str]:
    """Append the id to labels that several ids share (two configurations
    of one plugin side by side)."""
    counts: dict[str, int] = defaultdict(int)
    for label in labels.values():
        counts[label] += 1
    return {
        key: f"{label} ({key})" if counts[label] > 1 and label != key else label
        for key, label in labels.items()
    }


def _experiment_status(experiment_dir: Path) -> tuple[str, int | None]:
    """``(status, scheduled sample count)`` from ``run_status.json``."""
    status_path = experiment_dir / "run_status.json"
    if not status_path.is_file():
        return "incomplete", None
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown", None
    count = status.get("runnable_count")
    planned = count if isinstance(count, int) and not isinstance(count, bool) else None
    return ("complete" if status.get("success") is True else "failed"), planned


def _run_identity(config_path: Path | None) -> tuple[str, str]:
    match = _RUN_ID.search(config_path.name) if config_path else None
    if match is None:
        return "", ""
    day, time = match.group(2), match.group(3)
    started = f"{day[:4]}-{day[4:6]}-{day[6:]} {time[:2]}:{time[2:4]}"
    return match.group(1), started


def _limit(payload: dict[str, Any]) -> int | None:
    value = (payload.get("benchmark_config") or {}).get("limit")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _count_lines(path: Path, chunk: int = 16 << 20) -> int:
    """Newline-terminated records; a still-being-written tail is not counted."""
    count = 0
    with path.open("rb") as fp:
        while block := fp.read(chunk):
            count += block.count(b"\n")
    return count


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _preview(text: object, limit: int = _QUESTION_PREVIEW_CHARS) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else f"{value[: limit - 1]}…"


def summarize_run(
    index: int, payload: dict[str, Any], benchmark: str
) -> dict[str, Any]:
    """Compact summary of one ``runs.jsonl`` record for lists and the table."""
    audit = payload.get("audit") or {}
    transcript = payload.get("transcript") or []
    if isinstance(transcript, dict):  # large record, see `records.summary_fields`
        turn_count = transcript["count"]
        first_turn = transcript["first"] or {}
    else:
        turn_count = len(transcript)
        first_turn = transcript[0] if transcript else {}
    metrics = payload.get("system_metrics") or {}
    tags: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    records = audit.get("report_records") or []
    for record in records:
        passed = bool(record.get("passed"))
        for key, value in (record.get("tags") or {}).items():
            counts = tags[f"{key}={value}"]
            counts[0] += int(passed)
            counts[1] += 1
    duration_ms = _finite(metrics.get("run_duration_ms"))
    score_available = audit.get("score_available", True) is not False
    return {
        "i": index,
        "scenario_id": payload.get("scenario_id"),
        "trace_id": payload.get("trace_id"),
        "sample_index": payload.get("sample_index"),
        "identity": [
            payload.get("scenario_id"),
            payload.get("agent_index"),
            payload.get("memory_index"),
            payload.get("evaluator_index"),
            payload.get("sample_index"),
        ],
        "accuracy": audit.get("accuracy_score"),
        "success": audit.get("is_successful"),
        "score_available": score_available,
        "unavailable_reason": audit.get("score_unavailable_reason"),
        "passed": len(audit.get("passed_keys") or []),
        "failed": len(audit.get("failed_keys") or []),
        "records": len(records),
        "turns": turn_count,
        "question": _preview(first_turn.get("question")) if turn_count else "",
        "duration_s": duration_ms / 1000.0 if duration_ms is not None else None,
        "tokens": metrics.get("llm_total_tokens"),
        "cost": metrics.get("llm_total_cost"),
        "memory_ops": metrics.get("memory_operation_count"),
        "memory_errors": metrics.get("memory_operation_error_count"),
        "has_metrics": bool(metrics),
        "tags": dict(tags),
        "scores": sample_scores(benchmark, audit) if score_available else {},
    }


def _root_key(root: Path) -> str:
    """A short, stable directory name for one resolved results root."""
    return hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:16]


class ResultsCatalog:
    """Discovers experiments and serves table / run data for one results dir."""

    def __init__(self, results_dir: Path, *, cache_dir: Path | None = None) -> None:
        self.root = Path(results_dir).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Results directory not found: {self.root}")
        # A cache directory outside the results tree may serve several
        # results roots, whose experiment ids can coincide: keep one
        # subdirectory per root.
        self.cache_dir = (
            Path(cache_dir).expanduser() / _root_key(self.root)
            if cache_dir
            else self.root / CACHE_DIRNAME
        )
        self._lock = threading.Lock()
        self._index_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._indexes: dict[str, RunIndex] = {}
        self._index_errors: dict[str, str] = {}
        self._trace_indexes: dict[str, tuple[list[int], dict[str, Any]]] = {}
        self._queue: list[str] = []
        self._queue_cv = threading.Condition()
        self._worker: threading.Thread | None = None
        self.experiments: dict[str, Experiment] = {}
        self.refresh()

    # -- discovery -----------------------------------------------------------

    def refresh(self) -> None:
        experiments: dict[str, Experiment] = {}
        for runs_path in sorted(self.root.rglob(RUNS_JSONL_FILENAME)):
            if _skip(self.root, runs_path):
                continue
            experiment_dir = runs_path.parent
            configs = sorted(experiment_dir.glob("experiment_config.*.yaml"))
            payload: dict[str, Any] = {}
            if configs:
                loaded = yaml.safe_load(configs[-1].read_text(encoding="utf-8"))
                payload = loaded if isinstance(loaded, dict) else {}
            agent_config = payload.get("agent_config") or {}
            experiment_id = experiment_dir.relative_to(self.root).as_posix()
            config_path = configs[-1] if configs else None
            status, planned = _experiment_status(experiment_dir)
            run_id, started = _run_identity(config_path)
            run_dir = experiment_dir.parent.relative_to(self.root).as_posix()
            traces_path = experiment_dir / TRACES_JSONL_FILENAME
            benchmark, variant, scope, benchmark_label = _benchmark_identity(
                payload.get("benchmark_config") or {}
            )
            memory_config = payload.get("memory_system_config") or {}
            experiments[experiment_id] = Experiment(
                id=experiment_id,
                path=experiment_dir,
                name=str(payload.get("experiment_name") or experiment_dir.name),
                benchmark=benchmark,
                memory_system=_memory_id(memory_config),
                agent_model=str(agent_config.get("model_id") or ""),
                status=status,
                config_path=config_path,
                runs_path=runs_path,
                run_dir="" if run_dir == "." else run_dir,
                run_id=run_id,
                started=started,
                planned_samples=planned,
                limit=_limit(payload),
                variant=variant,
                scope=scope,
                memory_backend=str(
                    (memory_config.get("backend_config") or {}).get("type") or ""
                ),
                benchmark_label=benchmark_label,
                traces_path=traces_path if traces_path.is_file() else None,
            )
        for memory, label in memory_labels(experiments.values()).items():
            for experiment in experiments.values():
                if experiment.memory_system == memory:
                    experiment.memory_label = label
        self._count_records(experiments.values())
        with self._lock:
            # Runs still being written have grown since they were indexed.
            for experiment_id, index in list(self._indexes.items()):
                experiment = experiments.get(experiment_id)
                if experiment is None or index.signature != self._signature(experiment):
                    del self._indexes[experiment_id]
            self._index_errors.clear()
            self.experiments = experiments
            self._submetric_rows = self._load_submetric_rows()

    def _count_records(self, experiments: Iterable[Experiment]) -> None:
        """Set ``recorded_samples``, recounting only files that changed.

        Counting newlines is far cheaper than indexing, so a size-based choice
        of source run can be made before any run is indexed.
        """
        cache_path = self.cache_dir / "record_counts.json"
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = {}
        counts: dict[str, list[int]] = {}
        for experiment in experiments:
            stat = experiment.runs_path.stat()
            entry = cached.get(experiment.id)
            if entry is None or entry[:2] != [stat.st_size, stat.st_mtime_ns]:
                entry = [
                    stat.st_size,
                    stat.st_mtime_ns,
                    _count_lines(experiment.runs_path),
                ]
            counts[experiment.id] = entry
            experiment.recorded_samples = entry[2]
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(counts), encoding="utf-8")
        except OSError:
            pass

    def experiment(self, experiment_id: str) -> Experiment:
        experiment = self.experiments.get(experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)
        return experiment

    # -- submetrics ----------------------------------------------------------

    def _load_submetric_rows(self) -> list[dict[str, Any]]:
        """Rows of every ``submetrics.csv``, tagged with their experiment id.

        Only used for the per-experiment summary; the criterion table is
        computed from ``runs.jsonl`` so it also covers runs without a CSV.
        """
        by_location = {
            (experiment.path.parent, experiment.path.name): experiment
            for experiment in self.experiments.values()
        }
        rows: list[dict[str, Any]] = []
        for csv_path in sorted(self.root.rglob(SUBMETRIC_CSV_FILENAME)):
            if _skip(self.root, csv_path):
                continue
            with csv_path.open(encoding="utf-8", newline="") as fp:
                for row in csv.DictReader(fp):
                    experiment = by_location.get(
                        (csv_path.parent, row.get("experiment") or "")
                    )
                    if experiment is not None:
                        experiment.has_submetrics = True
                        row["experiment_id"] = experiment.id
                        rows.append(row)
        return rows

    def headline_rows(self, experiment_id: str) -> list[dict[str, Any]]:
        return [
            {
                key: row.get(key)
                for key in (
                    "scope",
                    "dimension",
                    "criterion",
                    "metric_name",
                    "aggregation",
                    "metric_value",
                    "unit",
                    "sample_count",
                    "excluded_count",
                )
            }
            for row in self._submetric_rows
            if row.get("experiment_id") == experiment_id
            and row.get("scope", "headline") == "headline"
        ]

    # -- runs.jsonl ----------------------------------------------------------

    @staticmethod
    def _signature(experiment: Experiment) -> list[int]:
        stat = experiment.runs_path.stat()
        return [_INDEX_VERSION, stat.st_size, stat.st_mtime_ns]

    def _cache_path(self, experiment: Experiment) -> Path:
        digest = hashlib.sha1(experiment.id.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{digest}.json"

    def run_index(self, experiment_id: str) -> RunIndex:
        experiment = self.experiment(experiment_id)
        with self._index_locks[experiment_id]:
            signature = self._signature(experiment)
            cached = self._indexes.get(experiment_id)
            if cached is not None and cached.signature == signature:
                return cached
            cache_path = self._cache_path(experiment)
            index = self._read_cached_index(cache_path, signature)
            if index is None:
                index = self._build_index(
                    experiment.runs_path, signature, experiment.benchmark
                )
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps(
                            {
                                "signature": signature,
                                "offsets": index.offsets,
                                "summaries": index.summaries,
                            }
                        ),
                        encoding="utf-8",
                    )
                except OSError:
                    pass  # read-only results dir: keep the in-memory index only
            self._indexes[experiment_id] = index
            return index

    @staticmethod
    def _read_cached_index(cache_path: Path, signature: list[int]) -> RunIndex | None:
        if not cache_path.is_file():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("signature") != signature:
                return None
            return RunIndex(signature, payload["offsets"], payload["summaries"])
        except (OSError, ValueError, KeyError):
            return None

    def indexed(self, experiment_id: str) -> RunIndex | None:
        """The index if it is loaded and current, without building it.

        An index built while its run was still being written is dropped, so
        the next request queues a rebuild instead of serving stale rows.
        """
        index = self._indexes.get(experiment_id)
        if index is None:
            return None
        experiment = self.experiments.get(experiment_id)
        try:
            current = experiment is not None and index.signature == self._signature(
                experiment
            )
        except OSError:
            current = False
        if not current:
            with self._lock:
                if self._indexes.get(experiment_id) is index:
                    del self._indexes[experiment_id]
            return None
        return index

    def index_error(self, experiment_id: str) -> str | None:
        return self._index_errors.get(experiment_id)

    def request_indexes(self, experiment_ids: Iterable[str]) -> None:
        """Build missing indexes on a background thread, in the given order."""
        with self._queue_cv:
            for experiment_id in experiment_ids:
                if (
                    experiment_id in self._indexes
                    or experiment_id in self._index_errors
                    or experiment_id in self._queue
                ):
                    continue
                self._queue.append(experiment_id)
            if self._queue and (self._worker is None or not self._worker.is_alive()):
                self._worker = threading.Thread(
                    target=self._index_worker, name="results-browser-index", daemon=True
                )
                self._worker.start()
            self._queue_cv.notify()

    def _index_worker(self) -> None:
        while True:
            with self._queue_cv:
                if not self._queue:
                    return
                experiment_id = self._queue.pop(0)
            try:
                self.run_index(experiment_id)
            except Exception as exc:  # noqa: BLE001 - shown as an unavailable cell
                self._index_errors[experiment_id] = repr(exc)
                print(
                    f"Could not index {experiment_id}: {exc!r}",
                    file=sys.stderr,
                )

    @staticmethod
    def _build_index(runs_path: Path, signature: list[int], benchmark: str) -> RunIndex:
        offsets: list[int] = []
        summaries: list[dict[str, Any]] = []
        with RecordFile(runs_path) as records:
            for span in records.spans():
                try:
                    payload = summary_fields(records, span)
                except ValueError:
                    continue  # truncated tail of a still-running job
                finally:
                    records.release(span)
                summaries.append(summarize_run(len(offsets), payload, benchmark))
                offsets.append(span.start)
        return RunIndex(signature, offsets, summaries)

    def _offset(self, experiment_id: str, run_index: int) -> int:
        offsets = self.run_index(experiment_id).offsets
        if not 0 <= run_index < len(offsets):
            raise IndexError(run_index)
        return offsets[run_index]

    def raw_run(
        self, experiment_id: str, run_index: int
    ) -> tuple[int, Iterator[bytes]]:
        """``(size, chunks)`` of one record exactly as saved, without loading it."""
        offset = self._offset(experiment_id, run_index)
        records = RecordFile(self.experiment(experiment_id).runs_path)
        span = records.record_at(offset)

        def chunks() -> Iterator[bytes]:
            try:
                yield from records.iter_bytes(span)
            finally:
                records.close()

        return span.size, chunks()

    def run(
        self,
        experiment_id: str,
        run_index: int,
        *,
        limits: DisplayLimits | None = _DEFAULT_LIMITS,
    ) -> dict[str, Any]:
        """One run record, bounded by `limits` (``None`` returns it verbatim).

        Records over ``LARGE_RECORD_BYTES`` are never loaded whole: they can
        only be shown bounded, built by walking the file in place.
        """
        offset = self._offset(experiment_id, run_index)
        with RecordFile(self.experiment(experiment_id).runs_path) as records:
            span = records.record_at(offset)
            if span.size > LARGE_RECORD_BYTES:
                if limits is None:
                    raise RecordTooLargeError(span.size)
                truncated = [True]
                walker = Walker(records.buffer, span.end)
                result = _bounded(walker, span, limits, truncated)
                return {"run": result, "truncated": True, "bytes": span.size}
            payload = json.loads(records.buffer[span.start : span.end])
        if limits is None:
            return {"run": payload, "truncated": False, "bytes": span.size}
        truncated = [False]
        result, _ = _truncate(payload, limits, truncated)
        return {"run": result, "truncated": truncated[0], "bytes": span.size}

    # -- execution traces ------------------------------------------------------

    def trace_index(self, experiment_id: str) -> dict[str, dict[str, Any]]:
        """``{trace_id: location}`` for an experiment's trace sidecar."""
        experiment = self.experiment(experiment_id)
        if experiment.traces_path is None:
            return {}
        with self._index_locks[f"traces:{experiment_id}"]:
            stat = experiment.traces_path.stat()
            signature = [_INDEX_VERSION, stat.st_size, stat.st_mtime_ns]
            cached = self._trace_indexes.get(experiment_id)
            if cached is not None and cached[0] == signature:
                return cached[1]
            cache_path = self._cache_path(experiment).with_suffix(".traces.json")
            traces: dict[str, Any] | None = None
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if payload.get("signature") == signature:
                    traces = payload["traces"]
            except (OSError, ValueError, KeyError):
                traces = None
            if traces is None:
                traces = index_traces(experiment.traces_path)
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps({"signature": signature, "traces": traces}),
                        encoding="utf-8",
                    )
                except OSError:
                    pass
            self._trace_indexes[experiment_id] = (signature, traces)
            return traces

    def _trace_location(
        self, experiment_id: str, run_index: int
    ) -> tuple[Path, str, int]:
        summaries = self.run_index(experiment_id).summaries
        if not 0 <= run_index < len(summaries):
            raise IndexError(run_index)
        trace_id = summaries[run_index].get("trace_id")
        experiment = self.experiment(experiment_id)
        if not trace_id or experiment.traces_path is None:
            raise TraceNotFoundError("this sample has no execution trace")
        location = self.trace_index(experiment_id).get(trace_id)
        if location is None:
            raise TraceNotFoundError(
                f"trace {trace_id} is missing from {TRACES_JSONL_FILENAME}"
            )
        return experiment.traces_path, trace_id, location["offset"]

    def trace(self, experiment_id: str, run_index: int) -> dict[str, Any]:
        """A sample's trace header and one timeline row per event."""
        path, _, offset = self._trace_location(experiment_id, run_index)
        return trace_overview(path, offset)

    def trace_event(
        self,
        experiment_id: str,
        run_index: int,
        position: int,
        *,
        limits: DisplayLimits = _DEFAULT_LIMITS,
    ) -> dict[str, Any]:
        """One event in full, bounded by `limits`."""
        path, _, offset = self._trace_location(experiment_id, run_index)
        truncated = [False]
        with RecordFile(path) as records:
            walker, span = event_span(records, offset, position)
            event = _bounded(walker, span, limits, truncated)
        return {"event": event, "truncated": truncated[0]}


class TraceNotFoundError(LookupError):
    """The sample has no trace, or its trace row is missing."""


class RecordTooLargeError(Exception):
    """A record too large to load whole; download it raw instead."""

    def __init__(self, size: int) -> None:
        super().__init__(f"record is {size:,} bytes")
        self.size = size


def _bounded(
    walker: Walker,
    span: Span,
    limits: DisplayLimits,
    flag: list[bool],
    *,
    collapsible: bool = False,
) -> Any:
    """`_truncate` for a value of a large record, walked without loading it."""
    if span.size <= limits.max_block_chars:
        value, _ = _truncate(walker.load(span), limits, flag, collapsible=collapsible)
        return value
    flag[0] = True
    first = walker.buffer[span.start]
    if first == ord('"'):
        preview = walker.text_preview(span, limits.max_chars)
        return f"{preview}\n… [{span.size:,} bytes, truncated]"
    if collapsible or first not in b"{[":
        preview = walker.text_preview(span, limits.block_preview_chars)
        return f"{preview}\n… [~{span.size:,} bytes of JSON truncated]"
    if first == ord("{"):
        return {
            key: _bounded(
                walker, child, limits, flag, collapsible=key in _COLLAPSIBLE_KEYS
            )
            for key, child in walker.members(span.start)
        }
    items: list[Any] = []
    skipped = 0
    for child in walker.items(span.start):
        if len(items) < limits.max_items:
            items.append(_bounded(walker, child, limits, flag))
        else:
            skipped += 1
    if skipped:
        items.append({"_truncated_items": skipped})
    return items


#: Keys whose (possibly huge) nested values may be collapsed to a preview.
_COLLAPSIBLE_KEYS = frozenset({"arguments", "output", "payload", "result"})


def _truncate(
    value: Any, limits: DisplayLimits, flag: list[bool], *, collapsible: bool = False
) -> tuple[Any, int]:
    """Return ``(bounded value, approximate serialized size)``."""
    if isinstance(value, str):
        if len(value) > limits.max_chars:
            flag[0] = True
            omitted = len(value) - limits.max_chars
            text = f"{value[: limits.max_chars]}\n… [{omitted:,} more characters truncated]"
            return text, len(text)
        return value, len(value)
    if isinstance(value, list):
        items = value[: limits.max_items]
        pairs = [_truncate(item, limits, flag) for item in items]
        result: Any = [item for item, _ in pairs]
        size = sum(item_size for _, item_size in pairs) + 2
        if len(items) < len(value):
            flag[0] = True
            result.append({"_truncated_items": len(value) - len(items)})
    elif isinstance(value, dict):
        result = {}
        size = 2
        for key, item in value.items():
            result[key], item_size = _truncate(
                item, limits, flag, collapsible=key in _COLLAPSIBLE_KEYS
            )
            size += item_size + len(key)
    else:
        return value, 8
    if collapsible and size > limits.max_block_chars:
        flag[0] = True
        preview = json.dumps(result, ensure_ascii=False)[: limits.block_preview_chars]
        text = f"{preview}\n… [~{size:,} characters of JSON truncated]"
        return text, len(text)
    return result, size


__all__ = [
    "CACHE_DIRNAME",
    "DisplayLimits",
    "Experiment",
    "RecordTooLargeError",
    "distinct_labels",
    "memory_labels",
    "ResultsCatalog",
    "TraceNotFoundError",
    "summarize_run",
]
