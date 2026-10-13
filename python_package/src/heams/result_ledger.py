"""Durable, keyed JSONL storage for completed benchmark runs."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from heams.results import RunResult

type ResultKey = tuple[str, int]


class ResultLedgerError(ValueError):
    """Raised when a result ledger cannot be trusted for resumption."""


def result_key(run: RunResult) -> ResultKey:
    """Return the logical identity of a completed scenario sample."""
    return run.scenario_id, run.sample_index


def expected_result_keys(
    scenario_ids: Iterable[str],
    *,
    num_samples: int = 1,
) -> set[ResultKey]:
    """Build the complete logical key set for materialized scenarios."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    return {
        (scenario_id, sample_index)
        for scenario_id in scenario_ids
        for sample_index in range(num_samples)
    }


class ResultLedger:
    """Append-only storage for validated ``RunResult`` records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._results: list[RunResult] = []
        self._keys: set[ResultKey] = set()
        self._closed = False
        self._load_existing()

    @property
    def results(self) -> list[RunResult]:
        """Return the validated records already present in the ledger."""
        return list(self._results)

    @property
    def completed_keys(self) -> set[ResultKey]:
        """Return the logical identities already present in the ledger."""
        return set(self._keys)

    def append(self, run: RunResult) -> None:
        """Persist one result before returning to the next unit of work."""
        key = result_key(run)
        if key in self._keys:
            raise ResultLedgerError(
                f"Duplicate result key {key!r} in ledger {self.path}"
            )

        if self._closed:
            raise ValueError(f"Result ledger {self.path} is closed")
        with self.path.open("a", encoding="utf-8") as ledger_file:
            ledger_file.write(run.model_dump_json() + "\n")
            ledger_file.flush()
            os.fsync(ledger_file.fileno())
        self._results.append(run)
        self._keys.add(key)

    def close(self) -> None:
        """Prevent further appends to the ledger."""
        self._closed = True

    def __enter__(self) -> ResultLedger:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return

        data = self.path.read_bytes()
        has_partial_line = bool(data) and not data.endswith(b"\n")
        durable_data = data
        if has_partial_line:
            durable_end = data.rfind(b"\n") + 1
            durable_data = data[:durable_end]

        for line_number, raw_line in enumerate(durable_data.splitlines(), start=1):
            if not raw_line.strip():
                raise ResultLedgerError(
                    f"Malformed empty run record at {self.path}:{line_number}"
                )
            try:
                run = RunResult.model_validate_json(raw_line)
            except (TypeError, ValueError) as exc:
                raise ResultLedgerError(
                    f"Malformed run record at {self.path}:{line_number}: {exc}"
                ) from exc

            key = result_key(run)
            if key in self._keys:
                raise ResultLedgerError(
                    f"Duplicate result key {key!r} at {self.path}:{line_number}"
                )
            self._results.append(run)
            self._keys.add(key)

        if has_partial_line:
            with self.path.open("r+b") as ledger_file:
                ledger_file.truncate(len(durable_data))
