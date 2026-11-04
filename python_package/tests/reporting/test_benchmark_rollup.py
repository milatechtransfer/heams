"""Memory labels read back from saved experiment configs."""

from heams.reporting.benchmark_rollup import memory_label_from_variant_payload


def test_a_loadable_config_is_labelled_by_its_id() -> None:
    payload = {"id": "baseline", "backend_config": {"type": "in_context"}}

    assert memory_label_from_variant_payload(payload) == "baseline"


def test_a_config_needing_a_missing_plugin_is_still_labelled_by_its_id() -> None:
    """Results stay readable, and distinguishable, without every plugin installed."""
    payload = {
        "id": "milamem_retention_rules",
        "backend_config": {"type": "not_installed"},
        "not_installed_wrapper": {},
    }

    assert memory_label_from_variant_payload(payload) == "milamem_retention_rules"


def test_without_an_id_the_label_falls_back_to_the_memory_type() -> None:
    payload = {"backend_config": {"type": "not_installed"}}

    assert memory_label_from_variant_payload(payload) == "not_installed"
