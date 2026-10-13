"""Every benchmark declares the report metadata the results browser reads."""

from __future__ import annotations

from typing import get_args

import pytest

from heams.benchmarks.factory import BenchmarkConfig
from heams.benchmarks.info import (
    BenchmarkInfo,
    ReportDimension,
    RunScope,
    benchmark_info,
    benchmark_infos,
    token_count,
)

CONFIG_CLASSES = get_args(get_args(BenchmarkConfig)[0])


@pytest.mark.parametrize("config_class", CONFIG_CLASSES, ids=lambda c: c.__name__)
def test_every_benchmark_config_declares_report_info(config_class: type) -> None:
    info = getattr(config_class, "report_info", None)

    assert isinstance(info, BenchmarkInfo), (
        f"{config_class.__name__} must set `report_info: ClassVar[BenchmarkInfo]`"
    )
    assert info.display_name.strip()
    assert info.dimension in (ReportDimension.PERFORMANCE, ReportDimension.PRIVACY)
    assert all(group and all(tag for tag in group) for group in info.criterion_tags)


def test_registry_covers_every_config_type() -> None:
    types = {
        config_class.model_fields["type"].default for config_class in CONFIG_CLASSES
    }

    assert set(benchmark_infos()) == types
    assert benchmark_info("not_a_benchmark") is None


def test_display_names_are_unique() -> None:
    names = [info.display_name for info in benchmark_infos().values()]

    assert len(names) == len(set(names))


def test_default_scope_follows_the_limit() -> None:
    info = benchmark_info("locomo")

    assert info.scope({"limit": None}) is RunScope.FULL
    assert info.scope({"limit": 3}) is RunScope.SAMPLE


def test_beam_splits_are_variants_sized_by_their_tokens() -> None:
    info = benchmark_info("beam")

    assert info.variant_of({"split": "10M"}) == "10M"
    assert info.label("10M") == "BEAM 10M"
    assert info.tokens("10M") == 10_000_000
    assert benchmark_info("locomo").variant_of({"split": "x"}) is None


def test_retention_policy_selections_are_partial_runs() -> None:
    info = benchmark_info("retention_policy")

    assert info.scope({"policy_ids": ["RP-01"]}) is RunScope.PARTIAL
    assert info.scope({"policy_ids": None, "limit": None}) is RunScope.FULL


@pytest.mark.parametrize(
    ("label", "tokens"),
    [("100K", 100_000), ("1M", 1_000_000), ("512", 512), ("big", None)],
)
def test_token_count(label: str, tokens: int | None) -> None:
    assert token_count(label) == tokens
