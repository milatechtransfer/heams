"""Iterable Retention Policy benchmark: load rows, yield one BenchmarkSample per policy."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from heams.benchmarks.base import BaseBenchmark, SkipReason
from heams.benchmarks.retention_policy.config import RetentionPolicyBenchmarkConfig
from heams.benchmarks.retention_policy.samples import (
    group_rows_by_policy,
    load_rows,
    policy_rows_to_sample,
    validate_row,
)
from heams.benchmarks.sample import BenchmarkSample


class RetentionPolicyBenchmark(BaseBenchmark):
    """One sample per policy: every fact under it, retained then recalled."""

    def __init__(self, config: RetentionPolicyBenchmarkConfig) -> None:
        self.config = config
        self.skipped: list[SkipReason] = []
        self._samples: list[BenchmarkSample] = []
        self._materialize()

    def _materialize(self) -> None:
        rows = load_rows(self.config.dataset_path)

        if self.config.policy_ids is not None:
            wanted = set(self.config.policy_ids)
            unknown = wanted - {row.get("policy_id") for row in rows}
            if unknown:
                raise ValueError(
                    f"policy_ids names ids absent from the dataset: {sorted(unknown)}"
                )
            rows = [row for row in rows if row.get("policy_id") in wanted]

        # Validated per row, not per policy group: one malformed row should
        # cost only itself, not every other valid fact under its policy.
        valid_rows: list[dict] = []
        skipped: list[SkipReason] = []
        for row in rows:
            try:
                validate_row(row)
            except ValueError as error:
                skipped.append(
                    SkipReason(
                        path=Path(str(row.get("id", "<unknown>"))), reason=str(error)
                    )
                )
                continue
            valid_rows.append(row)

        groups = group_rows_by_policy(valid_rows)
        policy_ids = list(groups)
        if self.config.limit is not None:
            # Caps the number of policy samples, not raw rows: a sample is now
            # a whole policy's fact collection, so slicing rows first could
            # hand a policy a partial, non-representative fact set instead of
            # just reducing how many policies run.
            policy_ids = policy_ids[: self.config.limit]

        self._samples = [
            policy_rows_to_sample(groups[policy_id], config=self.config)
            for policy_id in policy_ids
        ]
        self.skipped = skipped

    @property
    def samples(self) -> Sequence[BenchmarkSample]:
        return self._samples


__all__ = ["RetentionPolicyBenchmark"]
